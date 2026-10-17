#include "../experiments/d44c4babfb46912da6c024d6f6d57c8f89b5d479199aaa7bb235bf00874d6fb1/policy.cpp"
#include <cassert>
#include <memory>

struct Board {
 static constexpr int S=121;
 std::uint8_t terrain[S]{},structure[S]{},visible[S],alive[2]={1,1};
 std::int16_t owner[S];std::int64_t army[S]{},total[2]={400,200},land[2]={20,10},cities[2]={1,-1};
 ArenaObservation obs{11,11,2,0,300,0,terrain,structure,owner,army,visible,alive,total,land,cities,nullptr};
 Board(){std::fill(visible,visible+S,1);std::fill(owner,owner+S,-1);owner[0]=0;army[0]=100;structure[0]=2;}
 void refresh(Policy &p){p.planningStart=std::chrono::steady_clock::now();p.update(&obs);p.assetHistory();p.pressureMap();p.buildKingRelief();p.buildAttacks();p.baselineDanger=p.dangerAfter(nullptr,p.homeIncoming);}
};

int main(){
 {
  Board b;std::fill(b.terrain,b.terrain+Board::S,1);b.terrain[0]=0;b.army[0]=1;
  for(int c=12;c<=17;c++)b.terrain[c]=0;
  for(int c=12;c<=67;c+=11){b.terrain[c]=0;b.owner[c]=0;b.army[c]=1;}
  b.owner[12]=0;b.structure[12]=1;b.army[12]=80;b.owner[17]=1;b.army[17]=70;
  auto p=std::make_unique<Policy>(0,1,"{}");b.refresh(*p);p->buildCityControl();
  assert(p->civicCount==1&&p->civicNeed[0]==65&&p->civicBefore[0]==80);
  Policy::Program q;q.kind=1;q.root=12;q.goal=23;q.count=1;q.orders[0]={12,23,false};
  assert(p->cityControlCost(q)==0.); // A nearby returnable army still covers.
  q.count=5;q.goal=67;for(int k=0;k<5;k++)q.orders[k]={12+11*k,23+11*k,false};
  assert(p->cityControlCost(q)>1.); // A complete march gives up that cover.
  b.army[12]=160;b.refresh(*p);p->buildCityControl();q.orders[0].half=true;
  assert(p->cityControlCost(q)==0.); // A real independent garrison remains.
  b.army[12]=20;b.owner[13]=0;b.army[13]=25;b.army[23]=25;
  b.refresh(*p);p->buildCityControl();assert(p->civicBefore[0]==44);
  p->proposalCount=0;p->cityReliefPrograms();bool funded=false;
  for(int j=0;j<p->proposalCount;j++)if(p->proposals[j].kind==26){
   auto plan=p->proposals[j];assert(plan.count==2);assert(p->collectionArrival(plan,12)>=66);funded=true;
  }
  assert(funded); // Separate donors have to execute both real merge orders.
  b.army[23]=1;b.refresh(*p);p->buildCityControl();p->proposalCount=0;p->cityReliefPrograms();
  for(int j=0;j<p->proposalCount;j++)assert(p->proposals[j].kind!=26);
  b.terrain[15]=1;b.refresh(*p);p->buildCityControl();assert(p->civicCount==0);
 }

}
