"""Offline logistics diagnostics using only the recorded player's owned tiles.

The collection curve describes one executable, static shortest-path tree.
It ignores growth and enemy actions and is neither an optimal collection
bound nor a survival certificate. Every transfer consumes one own order.
"""
def owned_transport(obs, budgets=(0, 4, 8, 12), minimum_cut_tiles=8):
    root = obs.general_pos
    owner = obs.owner.ravel()
    army = obs.army.ravel()
    owned = {c for c in range(obs.size) if owner[c] == obs.player_id}

    def adjacent(c):
        y, x = divmod(c, obs.width)
        for yy, xx in ((y-1, x), (y, x+1), (y+1, x), (y, x-1)):
            if 0 <= yy < obs.height and 0 <= xx < obs.width:
                yield yy * obs.width + xx

    if root not in owned:
        return {"status": "general_not_owned"}
    parents = {root: -1}
    depth = {root: 0}
    order = [root]
    for c in order:
        for n in adjacent(c):
            if n in owned and n not in parents:
                parents[n] = c
                depth[n] = depth[c] + 1
                order.append(n)
    connected = set(parents)
    budget = max(budgets)
    values = {}
    for c in reversed(order):
        if depth[c] > budget:
            continue
        row = [int(army[c])] * (budget+1)
        for n in adjacent(c):
            if parents.get(n) != c or n not in values:
                continue
            for total in range(budget, 0, -1):
                row[total] = max(row[total], max(
                    row[total-used-1] + max(0, values[n][used]-1)
                    for used in range(total)))
        values[c] = row

    # Iterative articulation search: a narrow owned connection is different
    # from an impassable terrain bottleneck. Losing it may leave alternative
    # routes through non-owned land; those routes require additional orders.
    entered = {root: 0}
    low = {root: 0}
    dfs_parent = {root: -1}
    size = {root: 1}
    mass = {root: max(0, int(army[root])-1)}
    cut_size, cut_mass = {}, {}
    stack = [(root, iter(adjacent(root)))]
    while stack:
        c, neighbours = stack[-1]
        n = next(neighbours, None)
        if n is None:
            stack.pop()
            parent = dfs_parent[c]
            if parent >= 0:
                low[parent] = min(low[parent], low[c])
                size[parent] += size[c]
                mass[parent] += mass[c]
                if low[c] >= entered[parent]:
                    cut_size[parent] = cut_size.get(parent, 0) + size[c]
                    cut_mass[parent] = cut_mass.get(parent, 0) + mass[c]
            continue
        if n not in owned or n == dfs_parent[c]:
            continue
        if n in entered:
            low[c] = min(low[c], entered[n])
            continue
        entered[n] = low[n] = len(entered)
        dfs_parent[n] = c
        size[n], mass[n] = 1, max(0, int(army[n])-1)
        stack.append((n, iter(adjacent(n))))
    cuts = [{"cell": c, "guard": int(army[c]),
             "owned_tiles_cut_off": cut_size[c],
             "spare_army_cut_off": cut_mass[c]}
            for c in cut_size if c != root and cut_size[c] >= minimum_cut_tiles]
    cuts.sort(key=lambda c: (-c["spare_army_cut_off"],
                             -c["owned_tiles_cut_off"], c["cell"]))
    return {
        "status": "observed_owned_graph",
        "owned_tiles": len(owned), "home_connected_tiles": len(connected),
        "disconnected_spare_army": sum(max(0, int(army[c])-1)
                                       for c in owned-connected),
        "quiet_tree_home_force_by_orders": {str(k): values[root][k] for k in budgets},
        "critical_owned_connections": cuts[:4],
    }
