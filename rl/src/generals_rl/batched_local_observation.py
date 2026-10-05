"""Batch original local observations for trusted canonical 9x9 P2 state rows.

Original controllers and their reset/result lifecycle remain unchanged.
Extreme army rows use original safe stats.
"""
import numpy as np
import torch

from generals.protocol import STATE_DTYPES
from generals.core.batch import unpack_states
from generals.core.observation import Observation, observe, readonly
from generals.core.state import Action, PlayerStats, Structure, Terrain, UNKNOWN, UNKNOWN_OWNER, _readonly
from generals_rl.league_bridge import OriginalBaselineBridge


def local_observations(arrays, player_ids):
    """Return original types with independent arrays; inputs must be backend state.

    Bypass repeated GameState/MapLayout/PlayerStats validation only after typed
    batch guards. This is not an alternative parser for untrusted saved states.
    Public integer army/territory totals use int64, never weighted bincount.
    """
    count = len(player_ids)
    if not 1 <= count <= 256 or set(arrays) != set(STATE_DTYPES):
        raise ValueError("canonical 1..256 selected 9x9 P2 state rows required")
    for name, dtype in STATE_DTYPES.items():
        shape = (count, 81) if name in ("terrain", "structure", "owner", "army") else (
            (count, 2) if name in ("alive", "general_pos") else (count,))
        value = arrays[name]
        if (not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype)
                or value.shape != shape or not value.flags.c_contiguous):
            raise ValueError(f"{name} must be canonical contiguous {dtype} {shape}")
    pids = np.asarray(player_ids)
    if pids.shape != (count,) or pids.dtype != np.int64 or np.any((pids < 0) | (pids > 1)):
        raise ValueError("player_ids must be int64 absolute seats [selected lanes]")
    owner, army = arrays["owner"], arrays["army"]
    live = arrays["alive"].sum(axis=1)
    if (np.any(owner < -2) or np.any(owner > 1) or np.any(army < 0)
            or np.any(arrays["terrain"] > Terrain.MOUNTAIN)
            or np.any(arrays["structure"] > Structure.GENERAL)
            or np.any(arrays["general_pos"] < 0) or np.any(arrays["general_pos"] >= 81)
            or np.any(arrays["tick"] < 0) or np.any(live == 0)
            or np.any(arrays["terminated"] != (live == 1))
            or np.any(arrays["winner"] != np.where(live == 1, arrays["alive"].argmax(axis=1), -1))):
        raise ValueError("trusted backend state has invalid scalar/value domains")
    owned = (owner == pids[:, None]).reshape(count, 9, 9)
    padded = np.pad(owned, ((0, 0), (1, 1), (1, 1)))
    visible = np.zeros_like(owned)
    for dy in range(3):
        for dx in range(3):
            visible |= padded[:, dy:dy + 9, dx:dx + 9]
    terrain = np.where(visible, arrays["terrain"].reshape(count, 9, 9), UNKNOWN).astype(np.uint8)
    structure = np.where(visible, arrays["structure"].reshape(count, 9, 9), UNKNOWN).astype(np.uint8)
    local_owner = np.where(visible, owner.reshape(count, 9, 9), UNKNOWN_OWNER).astype(np.int16)
    local_army = np.where(visible, army.reshape(count, 9, 9), -1).astype(np.int64)
    # Conservative per-row bound: each player owns at most 81 nonnegative cells.
    # Do not even execute a potentially overflowing reduction for fallback rows.
    extreme = np.any((owner >= 0) & (army > np.iinfo(np.int64).max // 81), axis=1)
    safe = ~extreme
    totals = np.zeros((count, 2), dtype=np.int64)
    territory = np.zeros((count, 2), dtype=np.int64)
    cities = np.full((count, 2), -1, dtype=np.int64)
    for pid in (0, 1):
        cells = owner == pid
        totals[safe, pid] = np.where(cells[safe], army[safe], 0).sum(axis=1, dtype=np.int64)
        territory[:, pid] = cells.sum(axis=1, dtype=np.int64)
        own_rows = pids == pid
        cities[own_rows, pid] = (cells[own_rows] & (arrays["structure"][own_rows] == Structure.CITY)).sum(axis=1, dtype=np.int64)
    result = []
    for row, pid in enumerate(pids):
        if extreme[row]:
            original = unpack_states({name: value[row:row + 1] for name, value in arrays.items()}, width=9, height=9)[0]
            result.append(observe(original, int(pid), "local"))
            continue
        # Derived canonical int64 vectors have already passed the bound check.
        # Original PlayerStats type/backing-byte immutability, without 3 repeated
        # _integer_array validations/copies for every lane every tick.
        stats = object.__new__(PlayerStats)
        for name, values in (("army", totals), ("territory", territory), ("cities", cities)):
            object.__setattr__(stats, name, _readonly(values[row]))
        result.append(Observation(int(pid), int(arrays["tick"][row]),
            readonly(terrain[row]), readonly(structure[row]), readonly(local_owner[row]),
            readonly(local_army[row]), readonly(visible[row]), readonly(arrays["alive"][row]),
            int(arrays["general_pos"][row, pid]), stats, "local", bool(arrays["terminated"][row]),
            None if arrays["winner"][row] < 0 else int(arrays["winner"][row])))
    return result


class BatchedObservationBridge(OriginalBaselineBridge):
    """Original reset/result lifecycle; local batch construction defaults off."""
    def __init__(self, *args, batched_observations=False, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(batched_observations, bool):
            raise TypeError("batched_observations must be boolean")
        self.batched_observations = batched_observations

    def patch_actions(self, joint, batch, learner_ids):
        if not self.batched_observations:
            return super().patch_actions(joint, batch, learner_ids)
        if self.learner_seats is None or self._pending_actions is not None:
            raise RuntimeError("reset first and finish previous result before choosing actions")
        if (batch.players, batch.width, batch.height) != (2, 9, 9):
            raise ValueError("batched observations require 9x9 P2")
        if joint.shape != (batch.num_envs, 2) or joint.dtype != torch.long or joint.device != batch.device:
            raise ValueError("joint must be long [num_envs,2] on batch device")
        if learner_ids.dtype != torch.long or learner_ids.shape != (batch.num_envs,):
            raise ValueError("learner_ids must be long [num_envs]")
        seats = self._selected_host(learner_ids)
        if not np.array_equal(seats, self.learner_seats):
            raise RuntimeError("reset bridge after rotating learner seats")
        arrays = {name: self._selected_host(value) for name, value in batch.state.items()}
        observations = local_observations(arrays, 1 - seats)
        actions = [controller.act(obs) for controller, obs in zip(self.controllers, observations)]
        if any(not isinstance(action, Action) for action in actions):
            raise TypeError("original controllers must return original Action values")
        encoded = torch.tensor([a.encode() for a in actions], dtype=torch.long, device=batch.device)
        joint[self._rows(joint), torch.as_tensor(1 - seats, dtype=torch.long, device=batch.device)] = encoded
        self._pending_actions = actions
        return joint
