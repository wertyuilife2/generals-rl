"""Evaluation-only CUDA execution; training and simulator kernels stay unchanged."""
from __future__ import annotations

import torch
from tensordict import TensorDict

from .mdp import GameBatch


def pack_actions(actions, mask, action_space):
    """Check the normal action boundary with one eventual host transfer.

    Range and legality are carried beside the encoded action. Clamp only the
    diagnostic gather index so an invalid policy output still raises the same
    range error on the host rather than causing a device-side indexing error.
    """
    if actions.shape not in ((len(mask),), (len(mask), 1)):
        raise ValueError("actions must be [num_envs] or [num_envs, 1]")
    if actions.dtype not in (torch.int32, torch.int64):
        raise TypeError("encoded actions must be integer tensors")
    encoded = actions.reshape(-1).long()
    in_range = (encoded >= 0) & (encoded < action_space.size)
    legal = mask.gather(1, encoded.clamp(0, action_space.size - 1)[:, None]).squeeze(1)
    return torch.stack((encoded, in_range.long(), legal.long()), 1)


def unpack_actions(packed):
    values = packed.cpu().numpy()
    if not values[:, 1].all():
        raise ValueError("encoded action is out of range")
    if not values[:, 2].all():
        raise ValueError("action was masked at the joint decision boundary")
    return values[:, 0]


class FeaturePipeline:
    """One current observation per state, optionally replaying a fixed CUDA graph.

    Knowledge observes the tick's original active mask, including lanes ending
    on that tick. Graph warmup uses only the initial state: its repeated history
    and already-observed knowledge are invariant under another initial update.
    """

    def __init__(self, batch, player_ids, encoder, history, action_space, knowledge,
                 *, capture=False):
        self.ids, self.encoder, self.history = player_ids, encoder, history
        self.action_space, self.knowledge = action_space, knowledge
        self.batch = batch
        self.graph = None
        self.current = encoder.encode(batch, player_ids)
        self._observe()
        if capture and batch.device.type == "cuda":
            self.batch = GameBatch({name: value.clone() for name, value in batch.state.items()},
                                   batch.stats.clone(), batch.width, batch.height)
            self.active = torch.ones(batch.num_envs, dtype=torch.bool, device=batch.device)
            self.frames = history.frames
            stream = torch.cuda.Stream(device=batch.device)
            stream.wait_stream(torch.cuda.current_stream(batch.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._advance_graph()
            torch.cuda.current_stream(batch.device).wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                self._advance_graph()
            torch.cuda.current_stream(batch.device).wait_stream(stream)
            self.graph.replay()  # Capture allocates outputs but does not execute them.

    def _observe(self):
        values = {"policy": self.history.flattened(), "single_obs": self.current,
                  "action_mask": self.action_space.mask(self.batch, self.ids)}
        if self.knowledge is not None:
            values["knowledge"] = self.knowledge.encode(self.batch.state["tick"], self.ids).flatten(1)
        self.observations = TensorDict(values, batch_size=[self.batch.num_envs], device=self.batch.device)

    def _advance_graph(self):
        self.current = self.encoder.encode(self.batch, self.ids)
        self.frames.copy_(torch.cat((self.frames[:, 1:], self.current[:, None]), 1))
        if self.knowledge is not None:
            self.knowledge.update(self.batch, self.ids, mask=self.active)
        self._observe()

    def advance(self, batch, active):
        if self.graph is None:
            self.batch = batch
            self.current = self.encoder.encode(batch, self.ids)
            self.history.push(self.current)
            if self.knowledge is not None:
                self.knowledge.update(batch, self.ids, mask=active)
            self._observe()
        else:
            for name, value in batch.state.items():
                self.batch.state[name].copy_(value)
            self.batch.stats.copy_(batch.stats)
            self.active.copy_(active)
            self.graph.replay()


class GraphStudent:
    """Frozen snapshot logits with bounded power-of-two CUDA graph batches.

    Masking and WAIT/MOVE factorization remain the policy's original FP32 code.
    Each graph owns its inputs and output; the output is borrowed until its next
    replay on the same stream. No graph is shared across snapshots or training.
    """

    inference_execution = "cuda_graph_numeric_buckets"

    def __init__(self, policy):
        if policy.training:
            raise ValueError("CUDA graph inference requires an evaluation snapshot")
        self.policy = policy
        self.policy.requires_grad_(False)
        self.graphs = {}

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def _logits(self, observations):
        count = len(observations)
        if not count:
            return self.policy._forward(observations)
        bucket = 1 << (count - 1).bit_length()
        key = (bucket, str(observations["policy"].device),
               tuple((name, tuple(value.shape[1:]), value.dtype)
                     for name, value in observations.items()))
        if key not in self.graphs:
            static = TensorDict({name: value.new_zeros((bucket, *value.shape[1:]))
                                 for name, value in observations.items()},
                                batch_size=[bucket], device=observations.device)
            for name, value in observations.items():
                static[name][:count].copy_(value)
            # CUDA graph replay removes eager launch overhead. Avoid compiling a
            # second graph per bucket, whose cold cost dominates short evals.
            compiled = getattr(self.policy, "compile_forward", None)
            if compiled is not None:
                self.policy.compile_forward = False
            try:
                stream = torch.cuda.Stream(device=observations.device)
                stream.wait_stream(torch.cuda.current_stream(observations.device))
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self.policy._forward(static)
                torch.cuda.current_stream(observations.device).wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    logits = self.policy._forward(static)
                torch.cuda.current_stream(observations.device).wait_stream(stream)
            finally:
                if compiled is not None:
                    self.policy.compile_forward = compiled
            self.graphs[key] = static, graph, logits
        static, graph, logits = self.graphs[key]
        for name, value in observations.items():
            static[name][:count].copy_(value)
        graph.replay()
        return logits[:count]

    def action_probabilities(self, observations, *, validate_mask=True):
        logits = self._logits(observations)
        with torch.autocast(logits.device.type, enabled=False):
            return self.policy._probabilities_from_logits(
                logits, observations["action_mask"], validate_mask=validate_mask)

    def act_inference(self, observations):
        probabilities = self.action_probabilities(observations, validate_mask=False)
        if not self.policy.wait_move_factorization:
            return probabilities.argmax(-1, keepdim=True)
        wait = probabilities[:, :1] >= probabilities[:, 1:].sum(-1, keepdim=True)
        move = probabilities[:, 1:].argmax(-1, keepdim=True) + 1
        return torch.where(wait, torch.zeros_like(move), move)
