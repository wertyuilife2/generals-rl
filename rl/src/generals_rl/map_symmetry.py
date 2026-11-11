"""Square-map D4 permutations, applied to reset rows without enlarging the bank."""
import numpy as np
import torch


class MapSymmetry:
    def __init__(self, size, device=None):
        grid = np.arange(size * size).reshape(size, size)
        self.sources = np.stack([np.rot90(np.fliplr(grid) if flip else grid, k).ravel()
                                 for flip in (False, True) for k in range(4)])
        self.positions = self.sources.argsort(1)
        vectors = ((-1, 0), (0, 1), (1, 0), (0, -1))
        directions = []
        for flip in (False, True):
            for k in range(4):
                transformed = []
                for dy, dx in vectors:
                    dx = -dx if flip else dx
                    for _ in range(k):
                        dy, dx = -dx, dy
                    transformed.append(vectors.index((dy, dx)))
                directions.append(transformed)
        self.directions = np.asarray(directions)
        if device is not None:
            for name in ('sources', 'positions', 'directions'):
                setattr(self, name, torch.as_tensor(getattr(self, name), device=device))

    def apply(self, arrays, valid, transforms):
        """Transform spatial state only; seat identity, clocks and stats persist."""
        sources, positions = self.sources[transforms], self.positions[transforms]
        tensor = isinstance(sources, torch.Tensor)
        gather = (lambda value, indices: value.gather(1, indices)) if tensor else np.take_along_axis
        def take(value, indices):
            return gather(value, indices) if tensor else gather(value, indices, axis=1)
        result = dict(arrays)
        for key in ('terrain', 'structure', 'owner', 'army'):
            result[key] = take(arrays[key], sources)
        for key in ('general_pos', 'spawn_pos'):
            old = arrays[key]
            indices = old.long().clamp_min(0) if tensor else old.clip(min=0)
            mapped = take(positions, indices)
            result[key] = (torch.where(old >= 0, mapped, old).to(old.dtype) if tensor
                           else np.where(old >= 0, mapped, old).astype(old.dtype))
        return result, take(valid, sources)

    def actions(self, actions, transforms):
        """Map native action IDs for rule-equivalence checks and search clients."""
        tensor = isinstance(actions, torch.Tensor)
        encoded = (actions - 1).clamp_min(0) if tensor else (actions - 1).clip(min=0)
        rows = (torch.arange(len(actions), device=actions.device) if tensor else np.arange(len(actions)))[:, None]
        source = self.positions[transforms][rows, encoded // 8]
        direction = self.directions[transforms][rows, (encoded // 2) % 4]
        moved = 1 + source * 8 + direction * 2 + encoded % 2
        return torch.where(actions > 0, moved, actions) if tensor else np.where(actions > 0, moved, actions)
