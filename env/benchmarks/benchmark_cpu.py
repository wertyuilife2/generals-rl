"""Run from any working directory after installing generals-env."""
import argparse
import json
import platform
from time import perf_counter
import tracemalloc
import numpy as np
from generals_env import Action, CoreEngine, MapConfig, generate_map
from generals_env.controllers import make_controllers
from generals_env.runner import Runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=25)
    parser.add_argument('--players', type=int, default=4)
    parser.add_argument('--ticks', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--memory', action='store_true', help='measure Python allocations; slows timing')
    args = parser.parse_args()
    if args.ticks < 1: parser.error('ticks must be positive')
    if args.memory: tracemalloc.start()
    config = MapConfig(args.size, args.size, args.players)
    start = perf_counter()
    state = generate_map(config, args.seed)
    map_seconds = perf_counter() - start
    engine = CoreEngine(state.copy())
    # Preselected circular directions exercise validation, expansion, and fighting.
    moves = [[Action.move(int(pos), tick % 4) for pos in state.general_pos]
             for tick in range(args.ticks)]
    start = perf_counter()
    for actions in moves:
        if engine.state.terminated: break
        engine.step(actions)
    core_seconds = perf_counter() - start
    names = ['aggressive', 'expansion', 'defensive', 'random']
    kinds = [names[i % 4] for i in range(args.players)]
    runner = Runner(state.copy(), make_controllers(kinds, args.seed),
                    seed=args.seed, max_ticks=args.ticks)
    start = perf_counter()
    result = runner.run()
    ai_seconds = perf_counter() - start
    output = {'python': platform.python_version(), 'numpy': np.__version__,
              'platform': platform.platform(), 'processor': platform.processor(),
              'size': args.size, 'players': args.players, 'seed': args.seed,
              'map_seconds': map_seconds, 'core_ticks': engine.state.tick,
              'core_seconds': core_seconds, 'core_ticks_per_second': engine.state.tick/core_seconds,
              'ai_ticks': result['ticks'], 'ai_seconds': ai_seconds,
              'ai_ticks_per_second': result['ticks']/ai_seconds,
              'winner_id': result['winner_id'], 'stop_reason': result['stop_reason'],
              'trace_enabled': False, 'memory_tracking': args.memory}
    if args.memory:
        output['python_peak_bytes'] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    print(json.dumps(output, indent=2))


if __name__ == '__main__': main()
