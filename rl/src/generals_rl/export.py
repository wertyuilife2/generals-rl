"""Export saved learners into self-contained Arena policy bundles."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile

import torch

from generals_arena.policies.artifacts import RULES_PROTOCOL
from generals_arena.policies.artifacts import validate_bundle
from generals_arena.policies.code import load_policy_factory
from generals_arena.policies.code import policy_digest
from generals_arena.policies.code import snapshot_policy_code
from generals_arena.util import canonical
from generals_arena.util import file_digest


_FAMILIES = {
    'SpatialPPOPolicy': 'spatial', 'LightSpatialPPOPolicy': 'lightcnn',
    'InformationSpatialPPOPolicy': 'information',
    'OptimizedInformationSpatialPPOPolicy': 'optimized',
    'ScaledInformationSpatialPPOPolicy': 'scaled',
}
_FEATURES = dict(width=9, height=9, players=2, max_episode_steps=512, history_length=4,
                 productive_actions=False, combat_features=False,
                 king_threat_distance=False, persistent_knowledge=False)


def make_policy_spec(run_config):
    """Deployment options owned by RL, independent of Arena's manifest schema."""
    task, options = run_config['task'], run_config['runner']['policy']
    name = options['class_name'].rsplit('.', 1)[-1]
    if name not in _FAMILIES:
        raise ValueError(f'policy {name} has no deployable actor')
    if options.get('actor_obs_normalization', False):
        raise ValueError('export requires the supported explicit, unnormalized feature policy')
    features = {key: task.get(key, default) for key, default in _FEATURES.items()}
    network = {'family': _FAMILIES[name]}
    common = ('spatial_channels', 'residual_blocks', 'own_king_geometry', 'knowledge_channels',
              'input_channel_alignment', 'wait_move_factorization', 'amp_bfloat16')
    keys = (('backbone', 'model_size', 'spatial_mode', 'head_channels', 'input_channel_alignment',
             'wait_move_factorization', 'amp_bfloat16') if network['family'] == 'scaled' else common)
    network.update({key: options[key] for key in keys if key in options})
    if 'model_size' in network and isinstance(network['model_size'], str):
        network['model_size'] = network['model_size'].lower()
    network.setdefault('wait_move_factorization', network['family'] == 'scaled')
    network.setdefault('amp_bfloat16', False)
    if network['family'] in ('information', 'optimized'):
        network.setdefault('knowledge_channels', 0)
    model = {'codename': run_config.get('model', {}).get('codename') or run_config.get('run_id', 'policy'),
             'training_seed': task.get('seed', 42)}
    if run_config.get('run_id'):
        model['run_id'] = run_config['run_id']
    inference = {'mode': 'greedy', 'precision': 'from_config', 'execution': 'eager'}
    inference.update({key: run_config[key] for key in (
        'matmul_precision', 'cudnn_allow_tf32', 'cudnn_benchmark') if run_config.get(key) is not None})
    inference.update(run_config.get('inference', {}))
    return {'model': model, 'features': features, 'network': network, 'inference': inference}


def prepare_policy_source(run_config, output, *, source=None, policy_spec=None):
    """Snapshot runtime once per run; checkpoints reference these exact bytes."""
    live = Path(__file__).with_name('policy')
    source = Path(source) if source is not None else live
    source_id = policy_digest(source)
    if source_id != policy_digest(live):
        raise ValueError('resume policy source differs from this runtime; use its original code or warm-start')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    target = output/'policy_source'
    if target.exists():
        if policy_digest(target) != source_id:
            raise ValueError('run already owns a different policy source')
    else:
        snapshot_policy_code(source, target)
    current = make_policy_spec(run_config)
    if policy_spec is not None and any(policy_spec[key] != current[key]
                                       for key in ('features', 'network', 'inference')):
        raise ValueError('resume policy specification changed; use warm-start for a new policy')
    spec = deepcopy(policy_spec) if policy_spec is not None else current
    if policy_spec is None and not run_config.get('model', {}).get('codename'):
        spec['model']['codename'] = f"{spec['model']['codename']}-{output.name}"
    (output/'policy_spec.json').write_text(canonical(spec)+'\n')
    return {'source_path': str(target), 'source_id': source_id, 'policy_spec': spec}


def resolve_policy_source(checkpoint, info, *, source=None):
    """Locate pinned runtime after moving a complete run, never substitute live code."""
    if source is not None:
        return Path(source).resolve()
    recorded = info.get('source_path')
    if recorded and Path(recorded).is_dir():
        return Path(recorded).resolve()
    if info.get('source_id'):
        for parent in Path(checkpoint).resolve().parents:
            candidate = parent/'policy_source'
            if candidate.is_dir() and policy_digest(candidate) == info['source_id']:
                return candidate
        raise FileNotFoundError('checkpoint requires its pinned policy_source; restore it or pass --source')
    return Path(__file__).with_name('policy')


def export_policy(checkpoint, output, *, run_config=None, source=None,
                  completed_updates=None, codename=None, inference=None):
    """Export actor weights/config/source atomically, without optimizer or critic.

    New checkpoints pin their runtime in infos. Checkpoints without deployment metadata require
    their run_config and the selected policy source.
    mmap avoids materializing optimizer tensors from a full training checkpoint.
    """
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    saved = torch.load(checkpoint, weights_only=True, map_location='cpu', mmap=True)
    info = saved.get('infos') or {}
    if run_config is None and (checkpoint.parent/'run_config.json').is_file():
        run_config = json.loads((checkpoint.parent/'run_config.json').read_text())
    if isinstance(run_config, (str, Path)):
        run_config = json.loads(Path(run_config).read_text())
    spec = deepcopy(info.get('policy_spec'))
    if spec is None:
        if run_config is None:
            raise ValueError('checkpoint without deployment metadata requires its run_config to export')
        spec = make_policy_spec(run_config)
    source = resolve_policy_source(checkpoint, info, source=source)
    source_id = policy_digest(source)
    if info.get('source_id', source_id) != source_id:
        raise ValueError('checkpoint policy source does not match its recorded source_id')
    updates = completed_updates if completed_updates is not None else info.get('completed_ppo_updates')
    if updates is None:
        updates = int(saved.get('iter', -1))+1
    if type(updates) is not int or updates < 1:
        raise ValueError('export needs the number of completed PPO updates')
    if codename is not None:
        spec['model']['codename'] = codename
    spec['model'].update(completed_ppo_updates=updates, iteration=updates-1)
    if inference is not None:
        spec['inference'].update({'mode': inference} if isinstance(inference, str) else inference)
    name = f"{spec['model']['codename']}-{updates}"
    if output.exists():
        existing = validate_bundle(output)
        if (existing['name'] != name or json.loads(Path(existing['paths']['config']).read_text()) != spec
                or existing['manifest']['artifacts']['code']['sha256'] != source_id):
            raise ValueError('export destination belongs to a different policy')
        # The weights are checked too, so equal metadata cannot hide a different learner.
        actor = _actor_state(saved)
        previous = torch.load(existing['paths']['weights'], weights_only=True, map_location='cpu')
        if actor.keys() != previous['actor_state_dict'].keys() or any(
                not torch.equal(value, previous['actor_state_dict'][key]) for key, value in actor.items()):
            raise ValueError('export destination contains different actor weights')
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.export-', dir=output.parent) as temporary:
        staged = Path(temporary)/'bundle'
        staged.mkdir()
        torch.save({'actor_state_dict': _actor_state(saved)}, staged/'actor.pt')
        (staged/'model_config.json').write_text(canonical(spec)+'\n')
        snapshot_policy_code(source, staged/'policy')
        manifest = {'schema_version': 1, 'name': name, 'rules': RULES_PROTOCOL,
                    'entrypoint': 'runtime:create_controller',
                    'capabilities': {'maps': {'shapes': [[spec['features']['width'], spec['features']['height']]]},
                                     'player_counts': [spec['features']['players']]},
                    'artifacts': {
                        'weights': {'path': 'actor.pt', 'sha256': file_digest(staged/'actor.pt')},
                        'config': {'path': 'model_config.json', 'sha256': file_digest(staged/'model_config.json')},
                        'code': {'path': 'policy', 'sha256': source_id}}}
        (staged/'manifest.json').write_text(canonical(manifest)+'\n')
        factory = load_policy_factory(staged/'policy', manifest['entrypoint'], source_id)
        factory(weights=staged/'actor.pt', config_path=staged/'model_config.json', device='cpu')
        validate_bundle(staged)
        staged.rename(output)
    return output


def _actor_state(saved):
    if 'actor_state_dict' in saved:
        return {key: value.detach().cpu() for key, value in saved['actor_state_dict'].items()}
    values = {key.removeprefix('actor.'): value.detach().cpu()
              for key, value in saved['model_state_dict'].items() if key.startswith('actor.')}
    if not values:
        raise ValueError('checkpoint has no actor weights')
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, help='training run_config when checkpoint has no deployment metadata')
    parser.add_argument('--source', type=Path, help='explicit policy source directory')
    parser.add_argument('--updates', type=int)
    parser.add_argument('--codename')
    parser.add_argument('--mode', choices=('greedy', 'stochastic'))
    args = parser.parse_args(argv)
    bundle = export_policy(args.checkpoint, args.output, run_config=args.config, source=args.source,
                           completed_updates=args.updates, codename=args.codename, inference=args.mode)
    print(json.dumps({'bundle': str(bundle), **{key: validate_bundle(bundle)[key]
                                               for key in ('policy_id', 'name')}}, indent=2))


if __name__ == '__main__':
    main()
