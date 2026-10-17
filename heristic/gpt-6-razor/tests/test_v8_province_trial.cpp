#include "../experiments/f77ab97dd53ef9427c1a154fb8c0576026a56fb8106e6739b56e21e4d0ad59a4/policy.cpp"
#include <cassert>
#include <memory>
#include <iostream>
struct Board{
 static constexpr int N=121;
 uint8_t terrain[N],structure[N]{},visible[N],alive[2]={1,1};
 int16_t owner[N];int64_t army[N]{},total[2]={170,100},land[2]={10,10},cities[2]={0,0};
 ArenaObservation obs{11,11,2,0,134,0,terrain,structure,owner,army,visible,alive,total,land,cities,nullptr};
 Board(){std::fill(terrain,terrain+N,1);std::fill(visible,visible+N,1);std::fill(owner,owner+N,-1);
  for(int c=0;c<=10;c++){terrain[c]=0;owner[c]=0;army[c]=2;}structure[0]=2;army[0]=100;
  owner[10]=1;army[10]=43;
  for(int c:{17,18}){terrain[c]=0;owner[c]=0;army[c]=20;}
 }
 void setup(Policy &p){p.planningStart=std::chrono::steady_clock::now();p.update(&obs);p.assetHistory();p.pressureMap();p.buildAttacks();p.baselineDanger=p.dangerAfter(nullptr,p.homeIncoming);p.prepareProvinceRaids();p.proposalCount=0;}
};
int main(){
 Board b;auto p=std::make_unique<Policy>(0,1,"{}");b.setup(*p);
 assert(p->provinceRoot==10);assert(p->provinceCount>0);
 Policy::Program q;q.kind=28;q.root=q.goal=6;q.count=2;q.orders[0]={18,17,false};q.orders[1]={17,6,false};
 double value=p->provinceValue(q);std::cout<<"base "<<p->provinceBase<<" saved score "<<value<<"\n";assert(value>0);
 p->planningStart=std::chrono::steady_clock::now();p->provincePrograms();assert(p->proposalCount>0);
 auto selected=p->proposals[0];assert(selected.kind==28&&selected.count>=1);
 // The opposing raid can remove a donor before a delayed collection uses it.
 Policy::ProvinceRaid r;r.cell[0]=10;r.cell[1]=9;r.cell[2]=8;r.cell[3]=7;r.cell[4]=18;r.count=4;
 Policy::Program slow=q;slow.count=5;slow.orders[0]={0,1,true};slow.orders[1]={1,2,false};slow.orders[2]={2,3,false};slow.orders[3]={3,4,false};slow.orders[4]={18,17,false};
 bool feasible=true;p->provinceOutcome(slow,r,feasible);assert(!feasible);
 // A withdrawal cannot preserve the removed army as a blocking garrison.
 b.army[6]=40;b.army[17]=1;b.army[18]=1;b.setup(*p);
 Policy::ProvinceRaid straight;straight.count=6;for(int i=0;i<=6;i++)straight.cell[i]=10-i;
 Policy::Program wait;feasible=true;double held=p->provinceOutcome(wait,straight,feasible);assert(feasible);
 Policy::Program withdraw;withdraw.count=1;withdraw.orders[0]={6,17,false};feasible=true;
 double exposed=p->provinceOutcome(withdraw,straight,feasible);assert(feasible&&exposed>held);
 // Two independent large columns are outside the single-column model.
 b.owner[9]=1;b.army[9]=30;b.setup(*p);assert(p->provinceCount==0);
 {
  Board funded;funded.army[0]=70;funded.army[17]=funded.army[18]=1;
  auto agent=std::make_unique<Policy>(0,2,"{}");funded.setup(*agent);
  int original=agent->reserve[0];agent->planningStart=std::chrono::steady_clock::now();
  agent->provincePrograms();assert(agent->reserve[0]==original&&!agent->provinceCollecting);
  bool usesHome=false;
  for(int j=0;j<agent->proposalCount;j++)for(int k=0;k<agent->proposals[j].count;k++){
   auto x=agent->proposals[j].orders[k];
   if(x.s==0){assert(x.half);usesHome=true;}
  }
  assert(usesHome);
 }
}
