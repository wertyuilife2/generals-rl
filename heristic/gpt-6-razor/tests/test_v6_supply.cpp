// Pure planner invariants. This checks troop/order accounting, not a substitute
// game engine; actual games and decision timing are exercised through Arena.
#include "../v6/policy.cpp"
#include <cassert>
#include <memory>

int main(){
 std::uint8_t terrain[25],structure[25],visible[25],alive[2]={1,1};
 std::int16_t owner[25];std::int64_t army[25];
 std::fill(terrain,terrain+25,1);std::fill(structure,structure+25,0);
 std::fill(visible,visible+25,1);std::fill(owner,owner+25,-1);std::fill(army,army+25,0);
 for(int c:{12,7,2,11,10}){terrain[c]=0;owner[c]=0;army[c]=1;}
 army[2]=17;army[10]=20;structure[12]=2;
 // The lower edge is outside this seat's vision.
 for(int c=20;c<25;c++){terrain[c]=structure[c]=255;visible[c]=0;owner[c]=-3;army[c]=-1;}
 for(int c=4;c<25;c+=5){terrain[c]=structure[c]=255;visible[c]=0;owner[c]=-3;army[c]=-1;}
 std::int64_t totals[2]={40,10},land[2]={5,1},cities[2]={0,-1};
 ArenaObservation obs{5,5,2,0,100,12,terrain,structure,owner,army,visible,alive,totals,land,cities,nullptr};
 auto p=std::make_unique<Policy>(0,123,"{}");p->update(&obs);
 p->scheduleTree(12,4);
 for(int budget:{2,4}){
  Policy::Program program;p->extractSchedule(12,budget,program);
  assert(program.count==budget);
  int troops[25];for(int c=0;c<25;c++)troops[c]=std::max<std::int64_t>(0,army[c]);
  int edgeCount[25]{};
  for(int j=0;j<program.count;j++){
   auto x=program.orders[j];
   assert(owner[x.s]==0&&owner[x.t]==0&&p->direction(x.s,x.t)>=0);
   assert(!x.half&&troops[x.s]>1&&++edgeCount[x.s]==1);
   troops[x.t]+=troops[x.s]-1;troops[x.s]=1;
  }
  int total=0;for(int c=0;c<25;c++)total+=troops[c];
  assert(total==40&&troops[12]==p->schedule[12][budget]);
  if(budget==2){
   assert(troops[12]==20);
   assert(p->assemblyCredit(program,12,troops[12],200)==0);
  }else{
   assert(troops[12]==36);
   assert(p->assemblyCredit(program,12,troops[12],200)==16);
   assert(p->assemblyCredit(program,12,troops[12],25)==5);
  }
 }
}
