#include "../experiments/c8d78c77f2634a9a535770acdc8cc04c3a69844d26f987bb7e1b07205c5948e7/policy.cpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
struct Board {
 static constexpr int N=81;
 uint8_t terrain[N]{},structure[N]{},visible[N],alive[2]={1,1};
 int16_t owner[N];int64_t army[N]{},total[2]={30,180},land[2]={3,12},cities[2]{};
 ArenaObservation obs{9,9,2,1,200,0,terrain,structure,owner,army,visible,alive,total,land,cities,nullptr};
 Board(){std::fill(visible,visible+N,1);std::fill(owner,owner+N,-1);
  owner[0]=owner[40]=1;army[0]=100;army[40]=80;structure[0]=2;
  owner[80]=0;army[80]=2;structure[80]=2;
  owner[41]=-2;army[41]=10;structure[41]=1;
  owner[42]=owner[43]=0;army[42]=army[43]=1;
 }
 void setup(Policy &p){p.planningStart=std::chrono::steady_clock::now();p.update(&obs);p.assetHistory();p.pressureMap();p.intelligenceMap();p.supplyMap();p.buildKingRelief();p.buildAttacks();p.baselineDanger=p.dangerAfter(nullptr,p.homeIncoming);}
};
int main(){
 Board b;auto p=std::make_unique<Policy>(1,7,"{}");b.setup(*p);
 Policy::Program q;q.kind=1;q.routeModel=2;q.root=40;q.goal=43;q.count=3;q.born=200;
 q.orders[0]={40,41,false};q.orders[1]={41,42,false};q.orders[2]={42,43,false};
 double initial=p->remainingRouteValue(q);assert(initial>0);
 // The purchase is already complete. Its reward is absent from a suffix;
 // this is the same route as a fresh program in the same observed state.
 b.owner[41]=1;b.army[41]=69;b.army[40]=1;b.obs.tick=201;b.setup(*p);
 q.cursor=1;q.value=10000;double suffix=p->remainingRouteValue(q);
 Policy::Program fresh=q;fresh.orders[0]=q.orders[1];fresh.orders[1]=q.orders[2];fresh.cursor=0;fresh.count=2;
 double independent=p->remainingRouteValue(fresh);
 assert(suffix>0&&suffix<initial&&std::abs(suffix-independent)<1e-10);
 p->operationTasks[0].plan=q;p->operationTasks[0].progress=201;p->proposalCount=0;p->resumeOperation();
 assert(p->proposalCount==1&&std::abs(p->proposals[0].value-1.09*suffix)<1e-10);
 // A newly observed blocker at the later destination invalidates the whole
 // suffix even though the first order still takes its adjacent weak tile.
 b.army[43]=500;b.setup(*p);assert(p->remainingRouteValue(q)<-1e29);
 b.army[43]=1;b.setup(*p);
 // Revalidate reserves at the actual current source, not the old root.
 p->reserve[41]=20;assert(p->remainingRouteValue(q)>0&&q.orders[1].half);
 // Completed ordinary captures cannot keep paying on a fully owned suffix.
 b.owner[42]=b.owner[43]=1;b.setup(*p);assert(p->remainingRouteValue(q)<-1e29);
 std::cout<<"completed-city-not-paid-twice; suffix-equals-fresh; resume-reprices; future-blocker-rejected; current-reserve-honored; owned-suffix-no-capture-reward\n";
}
