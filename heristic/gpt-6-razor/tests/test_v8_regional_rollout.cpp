// Default: shared-response trial. Override the source to exercise another
// archived rollout. Define RAZOR_KEEP_ROLLOUT_PREFIX for the retained-prefix trial.
#ifndef RAZOR_POLICY_SOURCE
#define RAZOR_POLICY_SOURCE "../experiments/afd364b53d4292fc9d119e44bf79b7636ed21ceb06be5bad893c9e62219c19f5/policy.cpp"
#endif
#include RAZOR_POLICY_SOURCE
#include <cassert>
#include <memory>
#include <iostream>

struct Board {
 static constexpr int N=121;
 uint8_t terrain[N],structure[N]{},visible[N],alive[2]={1,1};
 int16_t owner[N];int64_t army[N]{},total[2]={200,240},land[2]={15,25},cities[2]={-1,2};
 ArenaObservation obs{11,11,2,1,301,0,terrain,structure,owner,army,visible,alive,total,land,cities,nullptr};
 Board(){std::fill(terrain,terrain+N,1);std::fill(visible,visible+N,1);std::fill(owner,owner+N,-1);}
 void tile(int c,int who,int n,int kind=0){terrain[c]=0;owner[c]=who;army[c]=n;structure[c]=kind;}
 void setup(Policy &p){
  p.planningStart=std::chrono::steady_clock::now();p.update(&obs);p.assetHistory();p.pressureMap();
  p.buildAttacks();p.baselineDanger=p.dangerAfter(nullptr,p.homeIncoming);
  p.baselineDanger=0; // Explicit local-model tests, not tactical acceptance.
  assert(p.prepareRegional());
 }
};
int main(){
 {
  Board b;b.tile(0,1,60,2);b.tile(1,1,1);b.tile(2,0,50);b.tile(11,1,1);b.tile(12,1,1,1);
  auto p=std::make_unique<Policy>(1,1,"{}");b.setup(*p);p->reserve[0]=1;
  Policy::Program q;q.kind=29;q.root=0;q.goal=11;q.count=1;q.orders[0]={0,11,false};
  auto v=p->regionalOutcome(q,0);assert(v< -900000.);
  assert(p->regionState.kingLost); // An initially blocked approach opens.
 }
 {
  Board b;for(int c=0;c<=10;c++)b.tile(c,1,1);b.tile(0,1,100,2);b.tile(5,1,7,1);b.tile(10,0,35);
  b.tile(16,1,30);b.tile(17,1,28);b.tile(6,1,1,1);
  auto p=std::make_unique<Policy>(1,2,"{}");b.setup(*p);p->regionalReset();
  int total=0;for(int c=0;c<b.N;c++)if(p->regionState.owner[c]==1)total+=p->regionState.army[c];
  int from=p->regionState.army[17];assert(p->regionalMove({17,16,false},1));
  int after=0;for(int c=0;c<b.N;c++)if(p->regionState.owner[c]==1)after+=p->regionState.army[c];
  assert(total==after&&p->regionState.army[17]==1);
  assert(p->regionState.army[16]==p->ready(16)+from-1);
  assert(!p->regionalMove({17,16,false},1)); // Already spent donor.
 }
 {
  Board b;b.tile(0,1,100,2);for(int c=1;c<=10;c++)b.tile(c,1,1);
  b.tile(5,1,4,1);b.tile(10,0,35);b.tile(16,1,4,1);b.tile(17,0,29);
  auto p=std::make_unique<Policy>(1,3,"{}");b.setup(*p);p->regionalReset();
  int troops[4],count=0;p->regionalSources(0,troops,count);assert(count==2);
  auto x=p->regionalEnemyOrder(0);assert(x.s==17&&x.t==16);
  assert(p->regionalMove(x,0));assert(p->regionState.owner[16]==0);
  // Re-evaluation selects another own asset after the first is occupied.
  x=p->regionalEnemyOrder(0);assert(x.s>=0&&x.t!=16);
 }
 {
  Board b;b.tile(0,1,100,2);for(int c=1;c<=10;c++)b.tile(c,1,1);
  b.tile(5,1,4,1);b.tile(10,0,35);b.tile(16,1,40);b.tile(17,1,31);
  auto p=std::make_unique<Policy>(1,4,"{}");b.setup(*p);
  Policy::Program wait;wait.root=0;
  double exposed=p->regionalOutcome(wait,0);
  Policy::Program cover;cover.root=5;cover.goal=5;cover.count=2;
  cover.orders[0]={17,16,false};cover.orders[1]={16,5,false};
  double held=p->regionalOutcome(cover,0);
  assert(held>-900000.&&exposed>-900000.);
  assert(p->regionState.owner[5]==1);
#ifdef RAZOR_KEEP_ROLLOUT_PREFIX
  Policy::Program realized;
  p->regionalOutcome(cover,0,&realized);
  assert(realized.count>=2&&realized.orders[0].s==17&&realized.orders[1].s==16);
  p->regionalOutcome(wait,0,&realized);
  assert(realized.count==0); // Do not compress future actions across a wait.
#endif
  // Current army at a completed post still gets a concrete next response.
  p->regionalReset();p->regionState.hero=16;
  auto x=p->regionalCoverOrder(16);assert(x.s==16&&x.t==5);
  Policy::Program sameWait; sameWait.root=16;
  assert(p->regionalOutcome(sameWait,0)==p->regionalOutcome(wait,0));
  // A terminal model margin cannot become a persistent legacy-auction bid.
  // The task remains available only for another regional comparison.
  auto &task=p->operationTasks[0];task.plan=cover;task.plan.kind=29;
  task.plan.value=1000000.;task.plan.born=p->tick;task.progress=p->tick;
  p->proposalCount=0;p->resumeOperation();assert(p->proposalCount==0);
  p->regionChoiceCount=0;p->regionalContinuations();assert(p->regionChoiceCount==1);
 }
 {
  Board b;b.obs.player=0;b.cities[0]=1;b.cities[1]=-1;
  b.tile(0,0,1,2);b.tile(1,0,60);b.tile(2,1,1,2);b.tile(11,1,100);b.tile(12,0,3,1);
  auto p=std::make_unique<Policy>(0,5,"{}");b.setup(*p);
  Policy::Program q;q.root=1;q.goal=2;q.count=1;q.orders[0]={1,2,false};
  double value=p->regionalOutcome(q,0);
  assert(value>900000.&&p->regionState.enemyKingLost&&!p->regionState.kingLost);
 }
 std::cout<<"shared ledger, pressure, two columns, terminal order and ongoing regional cover pass\n";
}
