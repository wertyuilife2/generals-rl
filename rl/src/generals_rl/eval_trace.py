"""Core-compatible evaluation traces from borrowed, batched host arrays.

Only the initial state is serialized through ``GameState``. Per-tick records
use the same byte ordering as ``core.recording.state_digest`` without building
state or layout objects, including when the simulator stores armies as int32.
"""
from hashlib import blake2b, sha256
import json

import numpy as np


_DIGEST_FIELDS = ("structure", "owner", "army", "alive", "general_pos")


def decode_joint_actions(encoded):
    """Return canonical ``[kind, source, direction, mode]`` action rows."""
    encoded = np.asarray(encoded)
    if encoded.ndim != 2 or encoded.shape[1] != 2 or encoded.dtype.kind not in "iu":
        raise ValueError("trace actions must be an integer [lanes, 2] array")
    if np.any(encoded < 0):
        raise ValueError("encoded trace actions cannot be negative")
    values = encoded.astype(np.int64, copy=False)
    if np.any(values < 0):
        raise ValueError("encoded trace actions exceed int64")
    move = values != 0
    remainder = np.maximum(values - 1, 0)
    decoded = np.empty((*values.shape, 4), dtype=np.int64)
    decoded[..., 0] = move
    decoded[..., 1] = np.where(move, remainder // 8, -1)
    decoded[..., 2] = (remainder // 2) % 4
    decoded[..., 3] = remainder % 2
    return decoded


def _topology_bytes(entry):
    """Cache exactly MapLayout's topology ID, using the initial terrain only."""
    cached = entry.get("topology_id_bytes")
    if cached is None:
        terrain = np.asarray(entry["initial_state"]["terrain"], dtype=np.uint8)
        digest = blake2b(digest_size=16)
        digest.update(b"generals-four-neighbor-v1\0")
        digest.update(np.asarray(terrain.shape, dtype="<i8").tobytes())
        digest.update(terrain.tobytes())
        cached = entry["topology_id_bytes"] = digest.hexdigest().encode()
    return cached


def state_digests_from_arrays(arrays, topology_ids, lanes):
    """Compute SHA256 bytes for selected host lanes in canonical core order.

    ``topology_ids`` contains ASCII topology-ID bytes, in the same order as
    ``lanes``. All five dynamic fields are converted and joined in one batch;
    the remaining per-lane work is hashing and the core JSON scalar suffix.
    A truncated game remains nonterminal, with winner ``None`` in that suffix.
    """
    lanes = np.asarray(lanes, dtype=np.intp)
    if lanes.ndim != 1 or len(topology_ids) != len(lanes):
        raise ValueError("trace lanes and topology IDs must have equal length")
    if not len(lanes):
        return np.empty((0, 32), dtype=np.uint8)
    values = np.concatenate([
        np.asarray(arrays[name][lanes], dtype="<i8").reshape(len(lanes), -1)
        for name in _DIGEST_FIELDS
    ], axis=1)
    # Explicit dtype also preserves canonical little-endian bytes on big-endian
    # hosts, on which concatenate may otherwise choose native byte ordering.
    values = np.asarray(values, dtype="<i8", order="C")
    raw = memoryview(values).cast("B")
    stride = values.shape[1] * values.itemsize
    ticks = np.asarray(arrays["tick"])[lanes]
    terminated = np.asarray(arrays["terminated"])[lanes]
    winners = np.asarray(arrays["winner"])[lanes]
    result = np.empty((len(lanes), 32), dtype=np.uint8)
    for index, topology in enumerate(topology_ids):
        digest = sha256()
        digest.update(topology)
        digest.update(raw[index * stride:(index + 1) * stride])
        winner = None if winners[index] < 0 else int(winners[index])
        digest.update(json.dumps([int(ticks[index]), bool(terminated[index]), winner]).encode())
        result[index] = np.frombuffer(digest.digest(), dtype=np.uint8)
    return result


def record_tick_arrays(candidates, specs, encoded_actions, arrays, lanes):
    """Append one joint tick to existing ``SelectedGameTraces`` candidates.

    ``lanes`` must include every game active immediately *before* the step,
    including games ending on this tick. Finished candidates are retained until
    ``finish_batch`` selects wins/losses in ascending game ID, so long games do
    not lose selection to faster ones. Arrays are borrowed and never retained.
    """
    selected = [(int(lane), candidates[specs[int(lane)].game_id])
                for lane in lanes if specs[int(lane)].game_id in candidates]
    if not selected:
        return
    indexes = np.asarray([lane for lane, _ in selected], dtype=np.intp)
    for lane, entry in selected:
        if int(arrays["tick"][lane]) != entry["ticks"] + 1 or entry["ticks"] >= len(entry["actions"]):
            raise RuntimeError("selected trace skipped or repeated a tick")
    actions = decode_joint_actions(np.asarray(encoded_actions)[indexes])
    digests = state_digests_from_arrays(arrays, [_topology_bytes(entry) for _, entry in selected], indexes)
    for offset, (_, entry) in enumerate(selected):
        index = entry["ticks"]
        entry["actions"][index] = actions[offset]
        entry["digests"][index] = digests[offset]
        entry["ticks"] += 1
