// Owning hypothetical CPU position for single-thread C++ planning.
// Populate a legal state before step(); after manual mutations set cache_valid=false.
// Copying creates an independent branch. No OpenMP or Python runtime is needed.
#pragma once
#include <array>
#include <vector>
#include "../backends/cpu/step_lane.h"
namespace generals::simulation {
using i64=std::int64_t;
struct Board {
    int w,h,p,n;
    std::vector<uint8_t> terrain,structure;
    std::vector<int16_t> owner;
    std::vector<i64> army;
    std::vector<uint16_t> growth;
    std::array<bool,8> alive{};
    std::array<int32_t,8> general{},deaths{},kills{},spawn{},owned_structures{};
    std::array<i64,8> leave{};
    std::array<i64,64> damage{};
    std::array<uint8_t,8> known{},capture{},warn{};
    i64 tick=0,idle=0,afk=0; int32_t winner=-1,reason=0,error=0,growth_count=0;
    bool terminated=false,cache_valid=false;
    std::array<i64,24> stats{},cached{};
    std::array<i64,72> moves{};
    std::array<i64,9> counters{};
    std::array<bool,2> done{};
    Board(int width,int height,int players):w(width),h(height),p(players),n(w*h),terrain(n),structure(n),owner(n,-1),army(n),growth(n){
        general.fill(-1);spawn.fill(-1);deaths.fill(-1);leave.fill(-1);
        for(int k=0;k<p;++k){alive[k]=true;known[k]=uint8_t(1<<k);}
    }
    // Bind after each copy; no pointer into the parent position survives a fork.
    generals::Batch batch(){return {terrain.data(),structure.data(),owner.data(),army.data(),alive.data(),general.data(),&tick,&terminated,&winner,&idle,&afk,&reason,leave.data(),deaths.data(),kills.data(),damage.data(),spawn.data(),known.data(),capture.data(),warn.data()};}
    void step(const std::array<i64,8>&actions){
        auto b=batch();
        generals::cpu::Outputs out{stats.data(),moves.data(),counters.data(),&error,done.data(),cached.data(),growth.data(),&growth_count,&cache_valid,owned_structures.data()};
        generals::cpu::step_lane(b,b,out,actions.data(),0,h,w,p,false,0);
    }
};
}
