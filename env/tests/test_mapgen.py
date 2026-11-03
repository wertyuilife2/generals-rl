"""Current official random-map constraints, with an independent flood fill."""
from collections import deque
import numpy as np
import pytest
from generals import MapConfig
from generals.core.mapgen import generate_map
from generals.core.recording import state_digest


def reachable(state,source,goal):
    h,w=state.owner.shape
    blocked=(state.layout.terrain==1)|(state.structure!=0)
    seen={source};queue=deque([source])
    while queue:
        c=queue.popleft()
        if c==goal:return True
        y,x=divmod(c,w)
        for dy,dx in ((-1,0),(0,1),(1,0),(0,-1)):
            yy,xx=y+dy,x+dx;other=yy*w+xx
            if 0<=yy<h and 0<=xx<w and other not in seen and (other==goal or not blocked[yy,xx]):
                seen.add(other);queue.append(other)
    return False


@pytest.mark.parametrize('players',range(2,9))
@pytest.mark.parametrize('density',[0,.5,1.])
def test_official_shapes_spawns_cities_and_connectivity(players,density):
    shapes=set()
    for seed in (0,17,2026):
        s=generate_map(MapConfig(players=players,mountain_density=density),seed);s.validate()
        h,w=s.owner.shape;shapes.add((w,h));base=16+players+max(0,players-6)*.5
        assert round(base)<=w<=round(base+4+players/2)
        assert round(base)<=h<=round(base+4+players/2)
        for a in s.general_pos:
            for b in s.general_pos:
                if a==b:continue
                assert abs(a//w-b//w)+abs(a%w-b%w)>max(9,9*(1.8-.1*players))
                assert reachable(s,int(a),int(b))
        cities=np.flatnonzero(s.structure.ravel()==1)
        assert len(cities)>0
        assert np.all((s.army.ravel()[cities]>=40)&(s.army.ravel()[cities]<=50))
        assert np.all(s.owner.ravel()[cities]==-2)
        if len(cities)>=players:
            for g in s.general_pos:assert min(abs(g//w-c//w)+abs(g%w-c%w) for c in cities)<=6
        assert np.all(s.army.ravel()[s.general_pos]==1)
        assert s.afk_turn==(71 if players==2 else 121)
    assert len(shapes)>1


def test_reproducibility_no_mutable_sharing_and_zero_densities():
    config=MapConfig(players=8,mountain_density=0,city_density=0)
    a,b=generate_map(config,982),generate_map(config,982)
    assert state_digest(a)==state_digest(b)
    assert not np.any(a.layout.terrain) and not np.any(a.structure==1)
    a.army.flat[a.general_pos[0]]=99
    assert b.army.flat[b.general_pos[0]]==1


def test_small_random_maps_fail_instead_of_changing_official_spacing():
    with pytest.raises(ValueError,match='spacing'):generate_map(MapConfig(7,7,2))


@pytest.mark.parametrize('options',[{'players':1},{'players':9},{'width':0},{'city_density':1.01},{'mountain_density':-.1}])
def test_invalid_distribution(options):
    with pytest.raises((ValueError,TypeError)):MapConfig(**options)
