"""Cached GPU BFS; independent programs for each player lane and target."""
import triton
import triton.language as tl


@triton.jit
def _update(terrain, generals, fog, valid, seats, rows, active, blocked_cache,
            goal_cache, distances, C: tl.constexpr, W: tl.constexpr,
            P: tl.constexpr, BLOCK: tl.constexpr):
    lane, target = tl.program_id(0), tl.program_id(1)
    if tl.load(active + lane):
        seat = tl.load(seats + lane)
        slot = tl.load(rows + lane) * P + seat
        field = slot * 2 + target
        cells = tl.arange(0, BLOCK)
        in_board = cells < C
        known = tl.load(terrain + slot*C + cells, in_board, other=1).to(tl.int32)
        present = tl.load(valid + lane*C + cells, in_board, other=0)
        hint = tl.load(fog + lane*C + cells, in_board, other=0)
        blocked = ~present | (known == 1) | ((known < 0) & hint)
        general = tl.load(generals + slot*C + cells, in_board, other=-1).to(tl.int32)
        marker = tl.where(target == 0, general == seat, (general >= 0) & (general != seat)) & ~blocked
        goal = tl.where(tl.sum(marker.to(tl.int32), 0) == 1,
                        tl.min(tl.where(marker, cells, C), 0), -1)
        old_blocked = tl.load(blocked_cache + field*C + cells, in_board, other=1)
        changed = tl.sum((blocked != old_blocked).to(tl.int32), 0) > 0
        if changed | (tl.load(goal_cache + field) != goal):
            tl.store(blocked_cache + field*C + cells, blocked, in_board)
            distance = tl.where((cells == goal) & ~blocked, 0, C)
            progressing = goal >= 0
            # A bounded for-loop avoids the Triton 3.4 while-layout compiler bug.
            # Stop relaxation after convergence; C-1 still covers winding paths.
            for iteration in range(C-1):
                if progressing:
                    up = tl.gather(distance, tl.maximum(cells-W, 0), 0)
                    down = tl.gather(distance, tl.minimum(cells+W, BLOCK-1), 0)
                    left = tl.gather(distance, tl.maximum(cells-1, 0), 0)
                    right = tl.gather(distance, tl.minimum(cells+1, BLOCK-1), 0)
                    near = tl.minimum(tl.where(cells >= W, up, C), tl.where(cells+W < C, down, C))
                    near = tl.minimum(near, tl.where(cells % W > 0, left, C))
                    near = tl.minimum(near, tl.where(cells % W + 1 < W, right, C))
                    updated = tl.where(blocked, C, tl.minimum(distance, near+1))
                    progressing = tl.sum((updated != distance).to(tl.int32), 0) > 0
                    distance = updated
            result = tl.where(goal < 0, -2, tl.where(distance < C, distance, -1))
            tl.store(distances + field*C + cells, result, in_board)
            tl.store(goal_cache + field, goal)


def update_distances(cache, terrain, general_owner, fog, valid, seats, rows, active):
    fog, valid, seats, rows, active = (x.contiguous() for x in (fog, valid, seats, rows, active))
    _update[(len(seats), 2)](terrain, general_owner, fog, valid, seats, rows, active,
        cache.blocked, cache.goals, cache.distances, cache.cells, cache.width,
        cache.players, triton.next_power_of_2(cache.cells), num_warps=4, num_stages=1)
