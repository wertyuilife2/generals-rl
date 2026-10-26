"""Arbitrary match plans share scalar records, seeds, replays and batch execution."""
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from generals import MapConfig
from generals_arena.matches.config import MatchConfig
from generals_arena.policies.entries import prepare_entrant
from generals_arena.matches.game import RecordedGame
from generals_arena.matches.game import close_game
from generals_arena.matches.replay import verified_replay
from generals_arena.matches.batch import execute_batch
from generals_arena.policies.entries import load_controller
from policy_test_utils import make_bundle


def entrants_and_plans(tmp_path, lineup, *, maps=2):
    players = len(lineup)
    entries, ids = {}, []
    for name in lineup:
        if name.startswith('package'):
            bundle = tmp_path/name
            if not bundle.exists():
                make_bundle(bundle, name=name, players=(players,))
            entry = prepare_entrant({'name':name,'kind':'package','bundle':str(bundle)})
        else:
            entry = prepare_entrant({'name':name,'kind':'baseline','policy':name})
        entries[entry['entrant_id']] = entry
        ids.append(entry['entrant_id'])
    config = asdict(MapConfig(9,9,players,.1,.05))
    plans = []
    for index in range(maps):
        for rotation in range(players):
            order = list(range(rotation,players)) + list(range(rotation))
            plans.append({'game_id':f'map{index}-rotation{rotation}', 'map':config, 'map_seed':701+index,
                'entrant_ids':[ids[instance] for instance in order], 'instance_ids':order,
                'controller_seeds':[2**48 + index*100 + instance for instance in order],
                'rating_group_size':players, 'custom_metadata':{'preserve':'unchanged'}})
    return entries, plans


def scalar_record(entries, plan, ticks):
    rules = MatchConfig(MapConfig(**plan['map']),max_ticks=ticks)
    controllers = [load_controller(entries[key],rules) for key in plan['entrant_ids']]
    game = RecordedGame(plan,SimpleNamespace(max_ticks=ticks),controllers)
    try:
        while not game.done:
            game.advance()
        return game.result(), game.replay_blob()
    finally:
        close_game(game)


@pytest.mark.parametrize('lineup', [
    ['package-a','random'],
    ['aggressive','random'],
    ['package-a','package-b','expansion'],
    ['package-a','package-a','random','defensive'],
])
def test_rotated_mixed_and_repeated_entrants_match_recorded_games(tmp_path,lineup):
    entries,plans = entrants_and_plans(tmp_path,lineup)
    batch,timing = execute_batch(entries,plans,max_ticks=24,record_replays=True)
    scalar,_ = execute_batch(entries,plans,max_ticks=24,record_replays=True,scalar=True)
    reference = [scalar_record(entries,plan,24) for plan in plans]
    assert batch == scalar == reference
    assert all(value >= 0 for value in timing.values())
    for record,blob in batch:
        assert record['custom_metadata'] == {'preserve':'unchanged'}
        assert len(list(verified_replay(blob,record))) == record['ticks']+1


def test_chunking_and_cached_runtime_keep_instance_seeds_and_output_order(tmp_path):
    entries,plans = entrants_and_plans(tmp_path,['package-a','package-b','random'])
    full,_ = execute_batch(entries,plans,max_ticks=10,record_replays=True)
    cache,chunked = {},[]
    try:
        for plan in plans:
            rows,_ = execute_batch(entries,[plan],max_ticks=10,record_replays=True,runtime_cache=cache)
            chunked.extend(rows)
        assert chunked == full
        assert len(cache) == 2
    finally:
        for controller in cache.values():
            close = getattr(controller,'close',None)
            if close is not None:
                close()


def test_cuda_matches_cpu_records_and_replay_bytes(tmp_path):
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    entries,plans = entrants_and_plans(tmp_path,['package-a','random'],maps=1)
    cpu,_ = execute_batch(entries,plans,max_ticks=18,record_replays=True)
    gpu,_ = execute_batch(entries,plans,max_ticks=18,record_replays=True,backend='gpu',device='cuda:0')
    assert gpu == cpu


def test_empty_batch_and_incompatible_instance_assignments(tmp_path):
    assert execute_batch({},[],max_ticks=1) == ([],{})
    entries,plans = entrants_and_plans(tmp_path,['package-a','random'],maps=1)
    plans[1]['instance_ids'] = [0,1]
    with pytest.raises(ValueError,match='instance assignment'):
        execute_batch(entries,plans,max_ticks=1)


@pytest.mark.parametrize('batched', [False,True])
def test_failed_controller_reset_closes_partial_group_and_drops_cache(monkeypatch,batched):
    import generals_arena.matches.batch as batch
    resources = []

    class Scalar:
        def __init__(self):
            self.closed = False
            self.number = len(resources)
            resources.append(self)
        def reset(self,seat,seed):
            if self.number == 1:
                raise RuntimeError('reset failure')
        def close(self):
            self.closed = True

    class Batched(Scalar):
        def reset_batch(self,seats,seeds):
            raise RuntimeError('reset failure')
        def act_batch(self,raw):
            raise AssertionError('never reached')

    monkeypatch.setattr(batch,'load_controller',lambda *args: (Batched if batched else Scalar)())
    cache = {}
    with pytest.raises(RuntimeError,match='reset failure'):
        batch._PolicyGroup({'entrant_id':'resource'},MatchConfig(),[0,1],[11,23],cache=cache,slot=0)
    assert len(resources) == (1 if batched else 2)
    assert all(resource.closed for resource in resources)
    assert not cache


def test_multiplayer_eliminated_seat_receives_no_active_feedback(monkeypatch,tmp_path):
    import numpy as np
    from generals import Action
    from generals.core.state import GameState,MapLayout
    import generals_arena.matches.batch as batch
    entries,plans = entrants_and_plans(tmp_path,['package-a']*3,maps=1)
    shape = (9,9)
    owner,structure,army = np.full(shape,-1,np.int16),np.zeros(shape,np.uint8),np.zeros(shape,np.int64)
    for seat,cell in enumerate((0,1,80)):
        owner.flat[cell],structure.flat[cell],army.flat[cell] = seat,2,1
    army.flat[0] = 20
    state = GameState(MapLayout(np.zeros(shape,np.uint8)),structure,owner,army,
                      np.ones(3,bool),np.array([0,1,80],np.int32))
    monkeypatch.setattr(batch,'generate_map',lambda *args: state.copy())
    feedback = []

    class Probe:
        def reset_batch(self,seats,seeds):
            self.seats = np.asarray(seats)
        def act_batch(self,raw):
            return np.where(raw['player_ids']==0,Action.move(0,1).encode(),0)
        def on_result_batch(self,raw):
            feedback.append(raw)

    monkeypatch.setattr(batch,'load_controller',lambda *args: Probe())
    records,_ = execute_batch(entries,[plans[0]],max_ticks=3)
    assert records[0][0]['final_alive'] == [True,False,True]
    dead = [row for row in feedback if row['player_ids'][0] == 1]
    assert len(dead) == 3 and all(not row['active'][0] for row in dead)
    alive = [row for row in feedback if row['player_ids'][0] != 1]
    assert all(row['active'][0] for row in alive)


@pytest.mark.parametrize('size',[0,1,3])
@pytest.mark.parametrize('kind',['numpy','list','cuda'])
def test_bad_batch_action_count_never_broadcasts_to_other_matches(monkeypatch,size,kind):
    import numpy as np
    from generals import Action
    import generals_arena.matches.batch as batch
    if kind == 'cuda':
        import torch
        if not torch.cuda.is_available():
            pytest.skip('CUDA unavailable')
        actions = torch.zeros(size,device='cuda',dtype=torch.int64)
    elif kind == 'list':
        actions = [Action.wait() for _ in range(size)]
    else:
        actions = np.zeros(size,dtype=np.int64)

    class Probe:
        def reset_batch(self,seats,seeds): pass
        def act_batch(self,raw): return actions

    monkeypatch.setattr(batch,'load_controller',lambda *args: Probe())
    group = batch._PolicyGroup({'entrant_id':'wrong-count'},MatchConfig(),[0,1],[11,23])
    with pytest.raises(ValueError,match='one action per match'):
        group.act({})


def test_cache_isolates_complete_map_context_and_reuses_matching_chunks(monkeypatch):
    import numpy as np
    import generals_arena.matches.batch as batch
    created = []

    class MapBound:
        def validate_rules(self,config,visibility):
            # A valid stateful controller may allocate map-specific state once.
            assert not hasattr(self,'width')
            self.width,self.visibility = config.width,visibility
        def reset_batch(self,seats,seeds):
            self.count = len(seats)
        def act_batch(self,raw):
            assert raw['width'] == self.width and raw['visibility'] == self.visibility
            return np.zeros(self.count,dtype=np.int64)

    def load(entry,rules):
        controller = MapBound()
        controller.validate_rules(rules.map,rules.visibility)
        created.append(controller)
        return controller

    monkeypatch.setattr(batch,'load_controller',load)
    entry,cache = {'entrant_id':'multi-map-policy'},{}
    original = MatchConfig(MapConfig(9,9,2,.1,.05))
    contexts = [original,MatchConfig(MapConfig(11,7,2,.1,.05)),
                MatchConfig(MapConfig(9,9,2,.15,.05))]
    groups = [batch._PolicyGroup(entry,rules,[0],[11],cache=cache,slot=0) for rules in contexts]
    repeated = batch._PolicyGroup(entry,original,[1],[23],cache=cache,slot=0)
    assert len(created) == len(cache) == 3
    assert repeated.controller is groups[0].controller
    assert len({id(group.controller) for group in groups}) == 3
    for group,rules in zip(groups,contexts,strict=True):
        assert group.act({'width':rules.map.width,'visibility':rules.visibility}).tolist() == [0]
