#include "../policy.cpp"
#include <cassert>
#include <iostream>
#include <memory>
struct Board {
 static constexpr int w=13,n=w*w,home=6*w+6;
 std::array<std::uint8_t,n> terrain{},building{},visible{},fog{};
 std::array<std::int16_t,n> owner{};
 std::array<std::int64_t,n> army{};
 std::array<std::uint8_t,2> alive{1,1},flags{};
 std::array<std::int64_t,2> total{30,51},land{1,2},cities{};
 std::array<double,2> departure{-1,-1};
 ArenaObservation view{w,w,2,0,100,home,terrain.data(),building.data(),owner.data(),army.data(),visible.data(),alive.data(),total.data(),land.data(),cities.data(),fog.data(),flags.data(),flags.data(),departure.data(),nullptr};
 Board(){visible.fill(1);owner.fill(-1);owner[home]=0;army[home]=30;building[home]=2;}
 void enemy(int c,int type=0){owner[c]=1;army[c]=25;building[c]=type;}
 std::unique_ptr<aegis::Agent> agent(int mode){auto a=std::make_unique<aegis::Agent>(0,1,("{\"deployment_mode\":"+std::to_string(mode)+"}").c_str());a->observe(&view);return a;}
};
int main(){int checks=0;
 {Board b;auto control=b.agent(0);
  for(int mode:{1,2}){auto a=b.agent(mode);assert(!a->deployment_active);assert(a->formation_forward==control->formation_forward);++checks;}}
 {Board b;b.enemy(6*b.w+11,2);
  for(int mode:{1,2}){auto a=b.agent(mode);assert(a->deployment_active);
   assert(a->formation_forward[6*b.w+9]>a->formation_forward[6*b.w+3]);
   assert(a->formation(6*b.w+9,30)>a->formation(6*b.w+3,30));++checks;}}
 {Board b;b.enemy(6*b.w+11);b.enemy(5*b.w+11);
  for(int mode:{1,2}){auto a=b.agent(mode);assert(a->deployment_active);
   assert(a->formation_forward[6*b.w+9]>a->formation_forward[6*b.w+3]);++checks;}}
 {Board b;b.enemy(6*b.w+11,2);for(int y=0;y<b.w;++y)b.terrain[y*b.w+8]=1;
  for(int mode:{1,2}){auto a=b.agent(mode);assert(a->formation_forward[b.home]==.25);assert(a->formation_forward[6*b.w+10]>.25);++checks;}}
 {Board x,y;x.enemy(6*x.w+11);y.enemy(6*y.w+11);
  int hidden=2*x.w+2;
  for(Board *b:{&x,&y}){b->visible[hidden]=0;b->fog[hidden]=1;b->terrain[hidden]=b->building[hidden]=255;b->owner[hidden]=-3;b->army[hidden]=-1;}
  y.owner[hidden]=1;y.army[hidden]=900;y.building[hidden]=2;y.terrain[hidden]=0;
  for(int mode:{1,2}){auto a=x.agent(mode),b=y.agent(mode);assert(a->formation_forward==b->formation_forward);assert(a->obstructed(hidden));++checks;}}
 {Board b;b.enemy(6*b.w+11,2);auto a=b.agent(2);assert(a->deployment_active);
  b.alive[1]=0;b.view.tick++;a->observe(&b.view);assert(!a->deployment_active);++checks;}
 {Board b;b.enemy(6*b.w+11,2);auto a=b.agent(2);a->search_until=std::chrono::steady_clock::time_point::min();b.view.tick++;a->observe(&b.view);
  assert(!a->deployment_active);for(int c=0;c<b.n;++c)assert(a->formation_forward[c]==.25+.75*std::min(1.,a->home_distance[c]/(.35*(b.w+b.w))));++checks;}
 // Independent reference: enumerate every actually remembered destination and
 // BFS its graph distance. Compare against the multi-source cost transform.
 int comparisons=0;
 for(int trial=0;trial<24;++trial){Board b;int king=10*b.w+10,city=2*b.w+10,plain=3*b.w+9;b.enemy(king,2);b.enemy(city,1);b.enemy(plain);
  for(int c=0;c<b.n;++c)if(c!=b.home && c!=king && c!=city && c!=plain && (c*17+trial*11)%23==0)b.terrain[c]=1;
  auto a=b.agent(1);std::vector<int> dk,dc,dp;a->distances({king},dk);a->distances({city},dc);a->distances({plain},dp);
  for(int c=0;c<b.n;++c)if(!a->obstructed(c)){
   double scale=.35*(b.w+b.w),v=std::max({std::exp(-dk[c]/scale),25./51*std::exp(-dc[c]/scale),1./51*std::exp(-dp[c]/scale)});
   assert(std::abs(a->formation_forward[c]-(.25+.75*v))<1e-10);++comparisons;
  }
 }
 std::cout<<checks<<" deployment scenarios; "<<comparisons<<" independent BFS comparisons passed\n";
}
