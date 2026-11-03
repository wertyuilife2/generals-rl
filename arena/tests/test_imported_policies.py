"""Protocol/timing regressions for the selected neko-dev policy imports."""
from pathlib import Path
import os
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
POLICIES = ['gpt-6-kestrel-v7', 'gpt-6-razor/v7', 'gpt-6-razor/v8']


@pytest.mark.parametrize('name', POLICIES)
def test_current_armies_fog_hints_and_moving_generals(tmp_path, name):
    razor = 'razor' in name
    source = ROOT/'heristic'/name/'policy.cpp'
    check = tmp_path/'check.cpp'
    check.write_text(f'''#include "{source}"
#include <cassert>
int main() {{
    std::uint8_t terrain[49]{{}}, structure[49]{{}}, visible[49], fog[49]{{}};
    std::int16_t owner[49]; std::int64_t army[49]{{}};
    std::fill(owner,owner+49,-1); std::fill(visible,visible+49,1);
    std::uint8_t alive[2]{{1,1}}, kill[2]{{}}, afk[2]{{}};
    std::int64_t total[2]{{1,5}}, land[2]{{1,1}}, cities[2]{{}};
    double neutralize[2]{{-1.,-1.}};
    owner[0]=0; owner[48]=1; structure[0]=structure[48]=2; army[0]=1;army[48]=5;
    terrain[24]=structure[24]=255;visible[24]=0;fog[24]=1;owner[24]=-3;army[24]=-1;
    ArenaObservation obs{{7,7,2,0,49,0,terrain,structure,owner,army,visible,alive,
        total,land,cities,fog,kill,afk,neutralize,nullptr}};
    auto p = new Policy(0,42{', "{}"' if razor else ''});
    p->update(&obs);
    assert(p->moveArmy(0)==0 && p->moveArmy(0,true)==0);
    assert(p->enemyGeneral[1]==48);
    {'assert(p->ter[24]==2 && p->grows(0,50)==2 && p->actionTurn(1)==49);' if razor else ''}
    obs.general_pos=1; obs.tick=50; owner[1]=0; army[1]=7; structure[1]=2; structure[0]=0;
    structure[48]=0; structure[47]=2; owner[47]=1; army[47]=3;
    terrain[24]=0;structure[24]=1;visible[24]=1;fog[24]=0;owner[24]=-2;army[24]=40;
    p->update(&obs);
    assert(p->moveArmy(1)==6 && p->moveArmy(1,true)==3);
    assert(p->gd[1]==0 && p->enemyGeneral[1]==47);
    assert(p->ter[24]==0 && p->st[24]==1);
    delete p;
}}
''')
    binary = tmp_path/'check'
    subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-O1',
                    '-I', str(ROOT/'arena/src/generals_arena/policies/sdk'),
                    str(check), '-o', str(binary)], check=True, capture_output=True, text=True)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
