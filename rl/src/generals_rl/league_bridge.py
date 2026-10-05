"""Sparse, episode-fixed original-controller bridge.

No task, historical pool, or global RNG is modified. Call reset after seat
rotation; patch_actions at the joint pre-growth boundary; on_result immediately
after native step and before reset or the next act. All host reads select rows
on the original device first; CPU/Core and GPUv1 contracts are verified.
"""
from __future__ import annotations

import numpy as np
import torch

from generals.controllers import AI_KINDS, make_controllers
from generals.core.batch import unpack_states
from generals.core.observation import observe
from generals.core.state import Action
from generals_rl.evaluate_builtin import decode_move


class OriginalBaselineBridge:
    """Overlay at most 256 fixed lanes; other lanes retain historical actions.

    The lane quota is explicit because original Python planning adds host cost.
    Keeping historical inference for overlaid lanes preserves its RNG sequence.
    """

    def __init__(self, lanes, *, kind="aggressive", seed=42):
        if kind not in AI_KINDS:
            raise ValueError("kind must be an original built-in controller")
        lanes = tuple(lanes)
        if (not 1 <= len(lanes) <= 256 or len(set(lanes)) != len(lanes)
                or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in lanes)):
            raise ValueError("provide 1..256 unique nonnegative integer fixed lanes")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.lanes = lanes
        self.kind, self.seed = kind, seed
        self.episode = np.full(len(lanes), -1, dtype=np.int64)
        self.learner_seats = None
        self.controllers = [None] * len(lanes)
        self._indices = {}
        self._pending_actions = None

    def _rows(self, tensor):
        key = str(tensor.device)
        if key not in self._indices:
            self._indices[key] = torch.tensor(self.lanes, device=tensor.device, dtype=torch.long)
        return self._indices[key]

    def _selected_host(self, tensor):
        # Selecting AFTER .cpu() would accidentally transfer all 16k lanes.
        return tensor.index_select(0, self._rows(tensor)).detach().cpu().numpy().copy()

    def controller_seed(self, offset):
        return int(np.random.SeedSequence(
            [self.seed, self.lanes[offset], int(self.episode[offset])]
        ).generate_state(1, dtype=np.uint64)[0])

    def reset(self, learner_ids, mask=None):
        if self._pending_actions is not None:
            raise RuntimeError("finish the joint tick result before resetting controllers")
        if learner_ids.dtype != torch.long or learner_ids.ndim != 1:
            raise ValueError("learner_ids must be long [num_envs]")
        if max(self.lanes) >= len(learner_ids):
            raise ValueError("fixed lane is outside this environment batch")
        seats = self._selected_host(learner_ids)
        if np.any((seats < 0) | (seats > 1)):
            raise ValueError("bridge requires two absolute seats")
        if self.learner_seats is None:
            if mask is not None:
                raise ValueError("first reset must initialize every selected lane")
            self.learner_seats = seats.copy()
        if mask is None:
            selected = np.ones(len(self.lanes), dtype=bool)
        else:
            if mask.dtype != torch.bool or mask.shape != learner_ids.shape:
                raise ValueError("reset mask must be bool [num_envs]")
            selected = self._selected_host(mask)
            if np.any(seats[~selected] != self.learner_seats[~selected]):
                raise ValueError("seat changed in a lane that was not reset")
        self._pending_actions = None
        for offset in np.flatnonzero(selected):
            self.episode[offset] += 1
            self.learner_seats[offset] = seats[offset]
            kinds = ["random", "random"]
            kinds[1 - seats[offset]] = self.kind
            self.controllers[offset] = make_controllers(
                kinds, seed=self.controller_seed(offset)
            )[1 - seats[offset]]

    def selected_states(self, batch):
        if (batch.players, batch.width, batch.height) != (2, 9, 9):
            raise ValueError("prototype requires 9x9 two-player states")
        arrays = {name: self._selected_host(value) for name, value in batch.state.items()}
        return unpack_states(arrays, width=batch.width, height=batch.height)

    def patch_actions(self, joint, batch, learner_ids):
        if self.learner_seats is None:
            raise RuntimeError("reset bridge before choosing actions")
        if self._pending_actions is not None:
            raise RuntimeError("on_result is required before the next decision")
        if (joint.shape != (batch.num_envs, 2) or joint.dtype != torch.long
                or joint.device != batch.device):
            raise ValueError("joint must be long [num_envs,2] on batch device")
        seats = self._selected_host(learner_ids)
        if not np.array_equal(seats, self.learner_seats):
            raise RuntimeError("reset bridge after rotating learner seats")
        states = self.selected_states(batch)
        actions = [controller.act(observe(state, int(1 - seat), "local"))
                   for controller, state, seat in zip(self.controllers, states, seats)]
        if any(not isinstance(action, Action) for action in actions):
            raise TypeError("original controllers must return original Action values")
        encoded = torch.tensor([action.encode() for action in actions],
                               dtype=torch.long, device=batch.device)
        # Only selected opponents are overwritten, never learner seats.
        opponent_seats = torch.as_tensor(1 - seats, dtype=torch.long, device=batch.device)
        joint[self._rows(joint), opponent_seats] = encoded
        self._pending_actions = actions
        return joint

    def on_result(self, result, submitted_joint):
        if self._pending_actions is None:
            raise RuntimeError("choose a joint action before its result")
        # Task must pass the exact submitted action, not just post-step state.
        moves = self._selected_host(torch.as_tensor(result.moves))
        actions = self._selected_host(torch.as_tensor(submitted_joint))
        for offset, controller in enumerate(self.controllers):
            opponent_seat = int(1 - self.learner_seats[offset])
            pending = self._pending_actions[offset]
            if actions[offset, opponent_seat] != pending.encode():
                raise RuntimeError("submitted opponent action differs from original controller")
            for pid in (0, 1):
                action = pending if pid == opponent_seat else Action.decode(int(actions[offset, pid]))
                move = decode_move(moves[offset, pid], pid, action)
                if move is not None and pid == opponent_seat:
                    controller.on_result(move)
                if (move is not None and move.eliminated == opponent_seat
                        and hasattr(controller, "eliminate")):
                    controller.eliminate()
        self._pending_actions = None
