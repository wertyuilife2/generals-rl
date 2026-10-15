#include "../v7/policy.cpp"
#include <cassert>
#include <memory>

struct Fixture {
 std::uint8_t terrain[25]{},structure[25]{},visible[25],alive[2]={1,1};
 std::int16_t owner[25];std::int64_t army[25]{},total[2]={300,100},land[2]={5,2},cities[2]={1,-1};
 ArenaObservation o{5,5,2,0,100,0,terrain,structure,owner,army,visible,alive,total,land,cities,nullptr};
 Fixture(){std::fill(visible,visible+25,1);std::fill(owner,owner+25,-1);owner[0]=0;army[0]=20;structure[0]=2;}
};
void originalSupplyChecks();
int main(){
 originalSupplyChecks();
 {
  Fixture f;std::fill(f.owner,f.owner+25,0);std::fill(f.army,f.army+25,1);
  f.army[0]=200;f.structure[12]=1;f.army[12]=100;
  f.owner[13]=f.owner[17]=1;f.army[13]=10;f.army[17]=20;
  f.total[0]=321;f.total[1]=30;f.land[0]=23;f.land[1]=2;
  auto p=std::make_unique<Policy>(0,123,"{}");p->update(&f.o);
  p->mission.kind=8;p->mission.value=6;p->mission.born=99;p->mission.root=7;p->mission.goal=13;
  p->mission.count=2;p->mission.cursor=1;p->mission.orders[0]={7,12,false};p->mission.orders[1]={12,13,false};
  p->missionTick=99;p->missionReceiptCell=12;p->missionReceiptForce=100;p->missionAction=arena_move(7,2,false);
  auto action=p->act(&f.o);
  assert(action>0&&((action-1)%2)==1); // Keep the city guarded on a resumed plan.
  assert((action-1)/8==12);
 }
 {
  Fixture f;f.owner[12]=f.owner[17]=f.owner[18]=0;f.structure[12]=1;
  f.army[12]=240;f.army[17]=1;f.army[18]=20;
  auto p=std::make_unique<Policy>(0,123,"{}");p->update(&f.o);
  Policy::Order split{12,17,true};auto action=arena_move(12,2,true);p->recordFlow(split,action);
  f.o.tick=101;f.army[12]=120;f.army[17]=121;
  ArenaMoveResult receipt{0,action,1,12,17,120,0,-1,1};f.o.previous=&receipt;p->update(&f.o);
  assert(p->sameFlow(12,17));
  Policy::Program collection;collection.orders[collection.count++]={17,12,false};
  int merged=p->ready(12)+p->ready(17)-1;
  assert(p->assemblyCredit(collection,12,merged,400)==0);
  // A genuinely independent donor still earns its marginal contribution.
  collection.orders[collection.count++]={18,17,false};
  merged+=p->ready(18)-1;
  assert(p->assemblyCredit(collection,12,merged,400)==19);
  // Failed action receipts must never manufacture shared lineage.
  auto fresh=std::make_unique<Policy>(0,123,"{}");f.o.tick=100;f.o.previous=nullptr;f.army[12]=240;f.army[17]=1;fresh->update(&f.o);fresh->recordFlow(split,action);
  f.o.tick=101;receipt.valid=0;f.o.previous=&receipt;fresh->update(&f.o);assert(!fresh->sameFlow(12,17));
 }
 {
  Fixture f;std::fill(f.terrain,f.terrain+25,1);std::fill(f.owner,f.owner+25,-1);std::fill(f.army,f.army+25,0);std::fill(f.structure,f.structure+25,0);
  for(int c:{10,11,12})f.terrain[c]=0;
  f.o.player=1;f.o.general_pos=12;f.owner[12]=1;f.structure[12]=2;f.army[12]=130;
  f.owner[10]=f.owner[11]=0;f.army[10]=f.army[11]=60;
  auto p=std::make_unique<Policy>(1,123,"{}");p->update(&f.o);p->buildAttacks();
  Policy::Order attack{12,11,false};
  assert(p->dangerAfter(&attack)>0); // The old static gate has the wrong order.
  assert(p->counterResponseSafe(attack,0,11,12,false));
  assert(p->counterResponseSafe(attack,0,10,11,false));
  f.army[10]=160;p->update(&f.o);p->buildAttacks();
  assert(!p->counterResponseSafe(attack,0,10,11,false));
 }
}

void originalSupplyChecks(){
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
