"""A normal single-thread policy can branch using the actual env CPU lane."""
from generals import Action, MapConfig
from generals.core.mapgen import generate_map
from generals.core.observation import observe
from generals_arena.policies import native


def test_policy_bundle_uses_shared_serial_simulation(tmp_path):
    source = tmp_path/'policy.cpp'
    source.write_text('''#include "arena_policy.h"
#include <generals_simulation.h>
extern "C" void *arena_create(int, std::uint64_t, const char *) { return new int(0); }
extern "C" void arena_destroy(void *p) { delete static_cast<int *>(p); }
extern "C" std::int64_t arena_act(void *, const ArenaObservation *) {
    generals::simulation::Board parent(9,9,2);
    parent.owner[0]=0;parent.army[0]=10;parent.structure[0]=2;
    parent.owner[80]=1;parent.army[80]=10;parent.structure[80]=2;
    parent.general[0]=parent.spawn[0]=0;parent.general[1]=parent.spawn[1]=80;
    auto child=parent;std::array<std::int64_t,8> actions{};
    actions[0]=arena_move(0,1,false);child.step(actions);
    if(child.error||child.tick!=1||child.owner[1]!=0||child.army[1]!=9||child.army[0]!=1)return -1;
    actions[0]=0;child.step(actions);
    if(child.tick!=2||child.army[0]!=2||child.army[80]!=11)return -1;
    if(parent.tick!=0||parent.army[0]!=10||parent.owner[1]!=-1)return -1;
    return 0;
}''')
    bundle = tmp_path/'policy.zip'
    native.build_bundle(source, bundle)
    controller = native.NativeController(bundle=bundle)
    try:
        controller.reset(0, 1)
        assert controller.act(observe(generate_map(MapConfig(15, 15, 2), 1), 0)) == Action(0)
    finally:
        controller.close()
