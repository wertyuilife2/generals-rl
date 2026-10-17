#include "../experiments/0428ed1d46ba3594420c76390b3e85c8a153e8223fa0c878235618f4cc61ddc2/policy.cpp"
#include <cassert>
#include <memory>

struct Board {
 static constexpr int S=121;
 std::uint8_t terrain[S],structure[S]{},visible[S],alive[2]={1,1};
 std::int16_t owner[S];std::int64_t army[S]{},total[2]={300,200},land[2]={30,50},cities[2]={1,-1};
 ArenaObservation obs{11,11,2,0,300,0,terrain,structure,owner,army,visible,alive,total,land,cities,nullptr};
 Board(){std::fill(terrain,terrain+S,1);std::fill(visible,visible+S,1);std::fill(owner,owner+S,-1);
  for(int c=0;c<=10;c++)terrain[c]=0;
  for(int c=0;c<=3;c++){owner[c]=0;army[c]=1;}army[0]=50;structure[0]=2;army[2]=30;structure[2]=1;owner[10]=1;army[10]=85;
 }
 void refresh(Policy &p){p.planningStart=std::chrono::steady_clock::now();p.update(&obs);p.assetHistory();p.pressureMap();p.buildAttacks();p.baselineDanger=p.dangerAfter(nullptr,p.homeIncoming);p.buildRegion();p.proposalCount=0;}
};
int main(){
 {
  Board b;auto p=std::make_unique<Policy>(0,1,"{}");b.refresh(*p);
  assert(p->regionEnemy==10&&p->regionCount==2&&!p->regionCovered);
  int reserve=p->reserve[0];assert(reserve>1);p->regionalPrograms();
  assert(p->reserve[0]==reserve);bool found=false;
  for(int j=0;j<p->proposalCount;j++)if(p->proposals[j].kind==27){
   auto q=p->proposals[j];assert(p->regionPlanSound(q));assert(q.count>=2);
   bool homeDonor=false;for(int k=0;k<q.count;k++)homeDonor|=q.orders[k].s==0&&q.orders[k].t==1&&!q.orders[k].half;
   assert(homeDonor);
   assert(p->regionTroops[0]<4); // Original home troops are not counted twice.
   assert(p->regionTroops[q.root]>=79);found=true;
  }
  assert(found);
  b.owner[1]=-1;b.army[1]=0;b.refresh(*p);p->regionalPrograms();assert(!p->proposalCount);
 }
 {
  Board b;b.army[0]=10;auto p=std::make_unique<Policy>(0,1,"{}");b.refresh(*p);p->regionalPrograms();
  assert(!p->proposalCount); // Public total army cannot fund the actual program.
 }
 {
  Board b;b.owner[8]=1;b.army[8]=80;auto p=std::make_unique<Policy>(0,1,"{}");b.refresh(*p);
  assert(p->regionEnemy<0); // Do not promise one response to two large columns.
 }
 {
  Board b;b.owner[10]=-1;b.army[10]=0;b.owner[3]=1;b.army[3]=85;
  auto p=std::make_unique<Policy>(0,1,"{}");b.refresh(*p);p->regionalPrograms();
  assert(!p->proposalCount); // No quiet collection certificate under immediate attack.
 }
 {
  Board b;b.army[0]=1;b.army[2]=90;auto p=std::make_unique<Policy>(0,1,"{}");b.refresh(*p);
  assert(p->regionCovered);
  Policy::Program q;q.kind=16;q.count=1;q.root=2;q.goal=3;q.orders[0]={2,3,false};
  assert(p->regionDepartureCost(q)==0.); // Nearby full move keeps the response funded.
  q.orders[0].half=true;assert(p->regionDepartureCost(q)>0.); // Two weak halves do not.
  b.owner[3]=-1;b.army[3]=0;b.refresh(*p);q.orders[0].half=false;
  assert(p->regionDepartureCost(q)==0.); // An actual new return edge is retained.
  b.army[2]=180;b.refresh(*p);q.orders[0].half=true;
  assert(p->regionDepartureCost(q)==0.); // A funded scouting half can depart.
 }
}
