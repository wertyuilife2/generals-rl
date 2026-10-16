#include "../experiments/c348e64f58be13e1fc31f4bd410476c5d64b7727accef4b853f5941379293b0a/policy.cpp"
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
  Board b;b.owner[59]=0;b.army[59]=80;b.owner[63]=1;b.army[63]=20;
  b.structure[61]=1;b.owner[61]=-2;b.army[61]=45;
  auto p=std::make_unique<Policy>(0,1,"{}");b.refresh(*p);
  auto &task=p->operationTasks[0];auto &q=task.plan;
  q.kind=22;q.root=59;q.goal=61;q.born=p->tick;q.count=2;
  q.orders[0]={59,60,false};q.orders[1]={60,61,false};task.progress=p->tick;
  assert(p->columnClaimed(59));
  p->assignMainColumn();p->intelligenceMap();p->commandPrograms();
  assert(p->commandColumn<0);
  // Ordinary one-tile raiding has no separate funded strategic objective.
  q.kind=1;q.goal=60;assert(!p->columnClaimed(59));
  // A completed plan releases the surviving force for a new operation.
  q.kind=22;q.goal=61;q.cursor=q.count;assert(!p->columnClaimed(59));
 }
 {
  Board b;b.owner[59]=0;b.army[59]=80;b.owner[63]=1;b.army[63]=20;
  for(int c=77;c<Board::S;c++){b.visible[c]=0;b.owner[c]=-3;b.army[c]=-1;b.terrain[c]=255;}
  auto p=std::make_unique<Policy>(0,1,"{}");b.refresh(*p);p->observeColumns();p->assignMainColumn();p->intelligenceMap();p->commandPrograms();
  assert(p->commandActive&&p->commandColumn==59&&p->proposalCount>0);
  auto q=p->proposals[p->proposalCount-1];assert(q.kind==26&&q.count>0);
  Policy::Program competing;q.cursor=0;competing.count=1;competing.kind=22;competing.goal=60;
  competing.orders[0]={59,58,false};
  // An economic generator cannot redirect the assigned column to a new
  // neutral purchase. Defence still owns the right to interrupt it.
  if(p->commandNext.t==58)competing.orders[0].t=48;
  assert(!p->commandAllows(competing));
  competing.kind=3;assert(p->commandAllows(competing));
  competing.kind=22;competing.orders[0].s=0;assert(p->commandAllows(competing));
  int minimum=p->commandMinimum;auto move=q.orders[0];
  auto action=arena_move(move.s,p->direction(move.s,move.t),move.half);
  p->recordCommand(q,move,action);
  int delivered=p->commandExpected,left=p->ready(move.s)-p->moveArmy(move.s,move.half);
  b.army[move.s]=left;b.owner[move.t]=0;b.army[move.t]=delivered;b.visible[move.t]=1;b.terrain[move.t]=0;b.obs.tick++;
  b.refresh(*p);p->observeCommand();
  assert(p->commandColumn==move.t&&p->commandMinimum==minimum&&p->commandFrom<0);
 }
 {
  Board b;for(int c=0;c<=66;c+=11)b.owner[c]=0;
  b.army[0]=5;b.army[55]=80;b.total[1]=240;b.land[1]=100;
  auto p=std::make_unique<Policy>(0,1,"{}");b.refresh(*p);
  p->commandColumn=55;p->exposedHomeNeed=56;
  assert(!p->homeCoveredWithoutColumn());
  // Another real column can take over the home obligation.
  b.owner[1]=0;b.army[1]=80;b.refresh(*p);p->exposedHomeNeed=56;
  assert(p->homeCoveredWithoutColumn());
  Policy::Program q;q.kind=6;Policy::Order x{55,44,false};
  p->recordCommand(q,x,arena_move(55,0,false));assert(p->commandReturn);
 }
 {
  Board b;b.owner[59]=b.owner[60]=0;b.army[59]=101;b.army[60]=0;b.structure[59]=1;
  auto p=std::make_unique<Policy>(0,1,"{}");b.refresh(*p);
  p->commandColumn=59;p->commandBorn=p->tick;p->commandMinimum=24;
  Policy::Program q;q.kind=1;Policy::Order x{59,60,true};
  p->commandActive=true;p->commandNext=x;
  p->recordCommand(q,x,arena_move(59,1,true));
  b.army[59]=51;b.army[60]=50;b.obs.tick++;b.refresh(*p);p->observeCommand();
  // The garrison retained at the city does not inherit the scout's role.
  assert(p->commandColumn==60);
 }
 {
  Board b;b.owner[58]=b.owner[59]=0;b.army[58]=80;b.army[59]=29;
  b.owner[63]=1;b.army[63]=20;
  auto p=std::make_unique<Policy>(0,1,"{}");b.refresh(*p);p->assignMainColumn();p->intelligenceMap();
  p->commandColumn=59;p->commandMinimum=52;p->commandNeed=80;p->commandBorn=p->tick;
  p->commandPrograms();assert(p->commandActive&&p->commandReserved[58]);
  auto q=p->proposals[p->proposalCount-1];
  assert(q.kind==26&&q.goal==59&&q.orders[0].s==58&&q.orders[0].t==59);
 }
 {
  Board b;std::fill(b.terrain,b.terrain+Board::S,1);b.terrain[0]=0;
  for(int c=55;c<66;c++)b.terrain[c]=0;
  b.owner[59]=0;b.army[59]=80;b.owner[60]=-2;b.structure[60]=1;b.army[60]=45;
  auto p=std::make_unique<Policy>(0,1,"{}");b.refresh(*p);
  p->taskRoutes(59,79,true,false,true);assert(p->opSteps[61]==INF);
  p->taskRoutes(59,79,true);assert(p->opSteps[61]==2);
 }
 {
  // A cleared scout does not erase the opponent's knowledge of our general.
  // Remote troop totals are not a deployable home reserve.
  Board b;b.army[0]=40;b.owner[90]=0;b.army[90]=90;
  b.total[1]=240;b.land[1]=100;b.owner[1]=1;b.army[1]=1;
  auto p=std::make_unique<Policy>(0,1,"{}");b.refresh(*p);
  assert(p->homeKnownTo[1]&&p->raidNeed==56);
  b.owner[1]=-1;b.army[1]=0;b.obs.tick=400;b.refresh(*p);
  assert(p->raidNeed==56&&p->reserve[0]>=56);
  // A nearby column on an owned return route supplies that same cover.
  b.owner[11]=0;b.army[11]=80;b.refresh(*p);
  assert(p->raidNeed==0&&p->reserve[0]<56);
  // The public prior is not carried over from an eliminated opponent.
  b.alive[1]=0;b.refresh(*p);assert(!p->homeKnownTo[1]);
 }

 {
  // Human replay pattern: the strongest available field column is below
  // the enemy's predicted concentration. That is no reason to split it into
  // two still weaker scouts; merging the departing half with another stack
  // remains possible.
  Board b;b.owner[59]=b.owner[60]=0;b.army[59]=77;b.army[60]=2;
  b.owner[63]=1;b.army[63]=110;
  auto p=std::make_unique<Policy>(0,1,"{}");b.refresh(*p);p->assignMainColumn();
  assert(p->mainColumn==59&&p->mainColumnNeed>38);
  assert(!p->scoutDepartureAllowed({59,60,true}));
  assert(p->scoutDepartureAllowed({59,60,false}));
  b.army[60]=30;b.refresh(*p);p->assignMainColumn();
  assert(p->scoutDepartureAllowed({59,60,true}));
 }
 {
  Board b;std::fill(b.terrain,b.terrain+Board::S,1);b.terrain[0]=0;
  for(int c=55;c<66;c++)b.terrain[c]=0;
  b.owner[55]=b.owner[56]=0;b.army[55]=100;b.army[56]=1;
  b.owner[65]=1;b.army[65]=60;
  auto p=std::make_unique<Policy>(0,1,"{}");b.refresh(*p);p->assignMainColumn();
  Policy::Program q;q.kind=16;q.root=55;q.goal=56;q.count=1;q.value=10.;q.orders[0]={55,56,true};
  // Two fifty-troop scouts do not replace the hundred-troop counter-column.
  assert(p->mainColumn==55&&p->mainColumnNeed>51);
  assert(p->programScore(q)<-1e29);
  q.orders[0].half=false;assert(p->programScore(q)>0.);
  // A split that joins another army and keeps a capable column is allowed.
  b.army[56]=40;b.refresh(*p);p->assignMainColumn();q.orders[0].half=true;
  assert(p->programScore(q)>0.);
 }
 {
  Board b;std::fill(b.terrain,b.terrain+Board::S,1);b.terrain[0]=0;
  for(int c=55;c<66;c++)b.terrain[c]=0;
  b.owner[55]=b.owner[56]=0;b.army[55]=b.army[56]=40;
  b.owner[58]=1;b.army[58]=60;
  auto p=std::make_unique<Policy>(0,1,"{}");b.refresh(*p);p->frontFormationPrograms();
  bool merge=false;for(int j=0;j<p->proposalCount;j++){
   auto q=p->proposals[j];if(q.count==1&&q.orders[0].s==55&&q.orders[0].t==56)merge=true;
  }
  assert(merge);
  b.owner[58]=-1;b.army[58]=0;b.owner[57]=1;b.army[57]=60;
  b.refresh(*p);p->proposalCount=0;p->frontFormationPrograms();assert(p->proposalCount>0);
  // The joined force must actually survive; proximity does not waive that.
  b.army[57]=100;b.refresh(*p);p->proposalCount=0;p->frontFormationPrograms();assert(p->proposalCount==0);
  // Rejoining a recent split is not the creation of another counter-column.
  b.army[57]=60;b.refresh(*p);p->flowId[55]=p->flowId[56]=1;
  p->flowAt[55]=p->flowAt[56]=p->tick;p->proposalCount=0;p->frontFormationPrograms();assert(p->proposalCount==0);
 }
 {
  Board b;std::fill(b.terrain,b.terrain+Board::S,1);b.terrain[0]=0;
  for(int c=55;c<66;c++)b.terrain[c]=0;
  b.owner[56]=0;b.army[56]=50;
  b.owner[60]=b.owner[61]=b.owner[62]=1;b.army[60]=75;b.army[61]=1;b.army[62]=80;b.structure[60]=1;
  auto p=std::make_unique<Policy>(0,1,"{}");b.refresh(*p);p->buildOpponentAssembly();
  assert(p->opponentAssembly[60]==154&&p->assemblyRequirement(56)>150&&p->assemblyRequirement(56)<200);
  // Proximity alone cannot transport an army across a mountain barrier.
  b.terrain[61]=1;b.owner[61]=-1;b.army[61]=0;b.refresh(*p);p->buildOpponentAssembly();
  assert(p->opponentAssembly[60]==75&&p->assemblyRequirement(56)<100);
 }
 {
  Board b;std::fill(b.terrain,b.terrain+Board::S,1);b.terrain[0]=0;
  for(int c=55;c<66;c++)b.terrain[c]=0;
  b.owner[58]=0;b.army[58]=80;
  b.owner[60]=b.owner[61]=b.owner[62]=1;b.army[60]=10;b.army[61]=1;b.army[62]=100;b.structure[60]=1;
  auto p=std::make_unique<Policy>(0,1,"{}");b.refresh(*p);p->buildCityRelief();
  Policy::Program q;q.kind=23;q.root=58;q.goal=60;q.count=2;
  q.orders[0]={58,59,false};q.orders[1]={59,60,false};
  // The later seat can finish its two-order relief by recapturing the city
  // in our arrival round, even though it cannot reinforce before our move.
  assert(p->assetRelief[60][2]==0);assert(!p->cityReplyHolds(q));
  b.army[58]=160;b.refresh(*p);p->buildCityRelief();assert(p->cityReplyHolds(q));
  // An army killed on the approach is not available a second time as relief.
  b.owner[58]=-1;b.army[58]=0;b.owner[63]=0;b.army[63]=150;
  b.refresh(*p);p->buildCityRelief();q.root=63;q.count=3;
  q.orders[0]={63,62,false};q.orders[1]={62,61,false};q.orders[2]={61,60,false};
  assert(p->cityReplyHolds(q));
 }

 {
  Board b;b.owner[59]=0;b.army[59]=80;b.owner[61]=1;b.army[61]=100;
  auto p=std::make_unique<Policy>(0,1,"{}");b.refresh(*p);
  assert(p->landingCanBeDenied({59,60,false}));
  b.owner[60]=0;b.army[60]=100;b.refresh(*p);
  assert(!p->landingCanBeDenied({59,60,false}));
  // Earlier-seat reinforcement can deny the capture before we even land.
  b.obs.player=1;b.owner[0]=b.owner[59]=1;b.owner[60]=b.owner[61]=0;b.army[60]=1;
  auto later=std::make_unique<Policy>(1,1,"{}");b.refresh(*later);
  assert(later->landingCanBeDenied({59,60,false}));
  // Hitting the source first can leave too few troops to buy a city, even
  // though that enemy is not strong enough to capture the source itself.
  b.owner[61]=-1;b.army[61]=0;b.owner[58]=0;b.army[58]=50;
  b.owner[60]=-2;b.structure[60]=1;b.army[60]=40;b.refresh(*later);
  assert(later->landingCanBeDenied({59,60,false}));
 }
 {
  Board b;b.owner[59]=0;b.army[59]=80;b.owner[60]=-2;b.army[60]=30;b.structure[60]=1;
  b.owner[65]=1;b.army[65]=100;
  auto p=std::make_unique<Policy>(0,1,"{}");b.refresh(*p);p->observeColumns();
  Policy::Program q;q.root=59;q.goal=60;q.kind=22;q.count=1;q.orders[0]={59,60,false};
  // The attacking column cannot count twice as the new garrison and its
  // returning support. A real, independent nearby donor changes the result.
  assert(p->capitalSecurity(q)<.8);
  b.army[60]=50;b.refresh(*p);q.value=1000.;
  // This purchase spends more army than all surviving local cover, and
  // that cover cannot match the nearby counter-column. Its rejection must
  // not depend on finding a higher-scoring optional alternative in time.
  assert(p->capitalSecurity(q)==0.&&p->programScore(q)<-1e29);
  b.army[60]=30;
  b.owner[49]=0;b.army[49]=80;b.refresh(*p);
  assert(p->capitalSecurity(q)>.99);
  // Spending that support on another city removes it from the final ledger.
  b.owner[48]=-2;b.structure[48]=1;b.army[48]=70;b.refresh(*p);
  q.count=2;q.orders[0]={49,48,false};q.orders[1]={59,60,false};
  assert(p->capitalSecurity(q)<.8);
 }
 {
  Board b;std::fill(b.terrain,b.terrain+Board::S,1);
  for(int c=0;c<11;c++){b.terrain[c]=0;b.owner[c]=0;b.army[c]=1;}
  for(int c=21;c<=120;c+=11){b.terrain[c]=0;b.owner[c]=0;b.army[c]=1;}
  b.army[0]=10;b.terrain[11]=0;b.owner[11]=0;b.army[11]=100;
  b.owner[120]=1;b.army[120]=140;
  auto p=std::make_unique<Policy>(0,1,"{}");b.refresh(*p);
  assert(p->gd[120]==20&&p->mobilizationETA[120]>12&&p->mobilizationETA[120]<=24);
  assert(p->mobilizationThreat[120]>b.army[0]);
  assert(p->reserve[0]<=b.army[0]);
  // A one-order reinforcement has plenty of slack. It should be available
  // as preparation, not immobilize home as a tactical emergency.
  assert(p->defend().source<0);
  // If assembling the required army is already impossible within that
  // deadline, begin usable reinforcement instead of waiting for radius 12.
  b.army[11]=40;b.refresh(*p);assert(p->defend().source==11);
  // Distant attack forecasts still pay the actual compulsory city guard.
  b.owner[10]=-2;b.structure[10]=1;b.army[10]=160;
  auto blocked=std::make_unique<Policy>(0,1,"{}");b.refresh(*blocked);
  assert(blocked->threat[120]==0&&blocked->mobilizationThreat[120]==0);
 }
 {
  Board b;b.owner[60]=b.owner[61]=b.owner[90]=1;
  b.structure[60]=2;b.structure[61]=1;b.army[60]=b.army[61]=100;b.army[90]=10;b.total[1]=250;
  auto p=std::make_unique<Policy>(0,1,"{}");b.refresh(*p);
  b.obs.tick++;b.visible[60]=b.visible[61]=0;b.owner[60]=b.owner[61]=-3;b.army[60]=b.army[61]=-1;
  b.army[90]=210;b.refresh(*p);
  // The visible 210 troops cannot also remain in two remembered 100-troop
  // garrisons when the opponent's entire public army is only 250.
  assert(p->rememberedDefense(60)+p->rememberedDefense(61)<=40);
  assert(p->rememberedDefense(90)==210);
  // Do not rewrite the observation history to make the estimate fit.
  assert(p->a[60]==100&&p->a[61]==100);
  // A newer sighting should not be diluted by a much older large garrison.
  b.obs.tick++;b.visible[60]=1;b.owner[60]=1;b.army[60]=20;b.total[1]=260;b.refresh(*p);
  b.obs.tick++;b.visible[60]=0;b.owner[60]=-3;b.army[60]=-1;b.total[1]=250;b.refresh(*p);
  assert(p->rememberedDefense(60)>=20);
  assert(p->rememberedDefense(60)+p->rememberedDefense(61)<=40);
 }
 {
  Board b;std::fill(b.terrain,b.terrain+Board::S,1);b.terrain[0]=0;
  for(int c=55;c<66;c++)b.terrain[c]=0;
  b.owner[55]=b.owner[56]=0;b.army[55]=b.army[56]=40;
  b.owner[57]=1;b.army[57]=1;
  auto p=std::make_unique<Policy>(0,1,"{}");b.refresh(*p);p->scheduleTree(55,1);
  assert(p->schedule[55][1]>=79);
  // A real adjacent attacking column can remove the donor before a long
  // collection finishes; relaxing the ownership shortcut must retain that.
  b.army[57]=100;b.refresh(*p);p->scheduleTree(55,1);
  assert(p->schedule[55][1]<0);
 }
 {
  Board b;b.owner[12]=1;b.army[12]=30;b.total[1]=180;b.land[1]=50;
  auto p=std::make_unique<Policy>(0,1,"{}");b.refresh(*p);
  // A reserve constraint alone cannot replace the troops consumed by a
  // successful defense. Rebuild it before there is another lethal attack.
  b.obs.tick+=20;b.owner[12]=0;b.army[12]=1;b.army[0]=2;
  b.owner[13]=1;b.army[13]=1;
  b.owner[1]=b.owner[2]=0;b.army[1]=25;b.army[2]=20;
  b.refresh(*p);p->readinessProgram();
  assert(p->proposalCount>0&&p->proposals[0].kind==6);
  b.army[0]=100;b.refresh(*p);p->proposalCount=0;p->readinessProgram();
  assert(p->proposalCount==0);
 }
 {
  Board b;std::fill(b.terrain,b.terrain+Board::S,1);b.terrain[0]=0;
  for(int c=55;c<66;c++)b.terrain[c]=0;
  b.land[0]=30;b.owner[55]=b.owner[56]=b.owner[57]=0;
  b.army[55]=b.army[56]=17;b.army[57]=1;
  b.owner[58]=-2;b.structure[58]=1;b.army[58]=25;
  for(int c=59;c<66;c++){b.visible[c]=0;b.terrain[c]=255;b.structure[c]=255;b.owner[c]=-3;b.army[c]=-1;}
  auto p=std::make_unique<Policy>(0,1,"{}");b.refresh(*p);p->intelligenceMap();
  Policy::Program q;q.kind=16;q.root=55;q.goal=59;q.count=5;
  q.orders[0]={56,55,false};q.orders[1]={55,56,false};q.orders[2]={56,57,false};
  q.orders[3]={57,58,false};q.orders[4]={58,59,false};
  // The approach reuses a donor tile, but its army is only collected once.
  // Both real donors are necessary to cross the known guard and reveal fog.
  assert(p->expeditionValue(q)>0);
  p->expeditionPrograms();assert(p->proposalCount>0);
  b.army[56]=1;b.refresh(*p);p->intelligenceMap();assert(p->expeditionValue(q)<-1e29);
  p->proposalCount=0;p->expeditionPrograms();assert(p->proposalCount==0);
  b.army[56]=17;b.refresh(*p);p->intelligenceMap();
  for(int c=59;c<66;c++)p->kingInformation[c]=0;
  assert(p->expeditionValue(q)<-1e29);
  // New information can invalidate the whole approach while its first
  // collection order is still legal. Do not continue into a known ambush.
  b.visible[60]=1;b.terrain[60]=0;b.structure[60]=0;b.owner[60]=1;b.army[60]=80;
  b.refresh(*p);p->intelligenceMap();assert(p->expeditionValue(q)<-1e29);
 }
 {
  Board b;std::fill(b.terrain,b.terrain+Board::S,1);b.terrain[0]=0;b.land[0]=30;
  for(int c=55;c<66;c++)b.terrain[c]=0;
  b.owner[55]=b.owner[56]=0;b.army[55]=80;b.army[56]=1;
  for(int c=59;c<66;c++){b.visible[c]=0;b.terrain[c]=255;b.structure[c]=255;b.owner[c]=-3;b.army[c]=-1;}
  auto p=std::make_unique<Policy>(0,1,"{}");b.refresh(*p);p->intelligenceMap();
  p->expeditionPrograms();assert(p->proposalCount>0);
  // Both modes can reveal the entire corridor. Keep the unneeded half
  // available rather than making an 80-troop scouting commitment.
  for(int j=0;j<p->proposalCount;j++)assert(p->proposals[j].orders[0].half);
 }
 {
  Board b;b.obs.tick=150;b.cities[0]=0;b.land[0]=30;
  std::fill(b.terrain,b.terrain+Board::S,1);b.terrain[0]=0;
  for(int c=55;c<66;c++)b.terrain[c]=0;
  b.owner[55]=b.owner[56]=0;b.army[55]=b.army[56]=40;
  b.owner[61]=1;b.army[61]=60;b.total[1]=110;b.land[1]=20;
  auto p=std::make_unique<Policy>(0,1,"{}");b.refresh(*p);p->observeColumns();
  p->mobilizationPrograms();assert(p->proposalCount>0);
  // A previously observed concentrated opponent remains a reason to build
  // a comparable field army after that column briefly leaves our vision.
  assert(p->observedColumnFraction[1]>.6);
  b.obs.tick++;b.visible[61]=0;b.owner[61]=-3;b.army[61]=-1;
  b.refresh(*p);p->observeColumns();assert(p->assemblyRequirement(56)>=60);
 }
 {
  Board b;std::fill(b.visible,b.visible+Board::S,0);
  std::fill(b.terrain,b.terrain+Board::S,255);std::fill(b.structure,b.structure+Board::S,255);
  std::fill(b.owner,b.owner+Board::S,-3);std::fill(b.army,b.army+Board::S,-1);
  b.visible[0]=1;b.terrain[0]=0;b.structure[0]=2;b.owner[0]=0;b.army[0]=100;
  b.terrain[120]=2;b.total[0]=b.total[1]=1000;
  auto p=std::make_unique<Policy>(0,1,"{}");b.refresh(*p);p->intelligenceMap();
  assert(p->informationPrice>0&&p->kingInformation[0]==0&&p->kingInformation[120]==0);
  double opening=p->eliminationValue(1);
  b.total[0]=b.total[1]=2000;b.refresh(*p);p->intelligenceMap();
  assert(p->eliminationValue(1)>opening);
  // Scouting a cell permanently removes it from the unknown-general mass.
  b.visible[60]=1;b.terrain[60]=0;b.structure[60]=0;b.owner[60]=-1;b.army[60]=0;
  b.obs.tick++;b.refresh(*p);p->intelligenceMap();assert(p->kingInformation[60]==0);
  b.visible[60]=0;b.terrain[60]=255;b.structure[60]=255;b.owner[60]=-3;b.army[60]=-1;
  b.obs.tick++;b.refresh(*p);p->intelligenceMap();assert(p->kingInformation[60]==0);
 }
 {
  Board b;std::fill(b.terrain,b.terrain+Board::S,1);b.terrain[0]=0;
  for(int c=55;c<66;c++)b.terrain[c]=0;
  b.owner[58]=b.owner[59]=0;b.army[58]=80;b.army[59]=30;
  b.owner[60]=b.owner[61]=1;b.army[60]=4;b.army[61]=70;b.structure[60]=1;
  auto p=std::make_unique<Policy>(0,1,"{}");b.refresh(*p);p->sources={58,59};
  p->assetFlips[60]=4;p->assetLostAt[60]=300;p->occupationPrograms();
  assert(p->proposalCount>0);
  for(int j=0;j<p->proposalCount;j++){
   auto &q=p->proposals[j];assert(q.count>=2&&p->mine(q.orders[0].t));
  }
  // A cheap recapture is not an occupation when the same visible army will
  // immediately take it back. The plan must first fund the retained force.
  b.army[58]=1;b.refresh(*p);p->proposalCount=0;p->occupationPrograms();
  assert(p->proposalCount==0);
 }
 {
  Board b;std::fill(b.terrain,b.terrain+Board::S,1);b.terrain[0]=0;
  for(int c=55;c<66;c++)b.terrain[c]=0;
  b.owner[55]=b.owner[56]=0;b.army[55]=b.army[56]=40;
  b.owner[61]=1;b.army[61]=60;
  auto p=std::make_unique<Policy>(0,1,"{}");b.refresh(*p);p->frontFormationPrograms();
  assert(p->proposalCount>0);bool combined=false;
  for(int j=0;j<p->proposalCount;j++){
   auto &q=p->proposals[j];if(q.kind==25&&q.root==56&&q.count==1&&q.orders[0].s==55)combined=true;
  }
  assert(combined);
  b.army[55]=1;b.army[56]=100;b.refresh(*p);p->proposalCount=0;p->frontFormationPrograms();
  // Relocating an existing superior column must not earn the creation bonus.
  assert(p->proposalCount==0);
 }
 {
  Board b;std::fill(b.terrain,b.terrain+Board::S,1);b.terrain[0]=0;
  for(int c=55;c<66;c++)b.terrain[c]=0;
  b.owner[56]=b.owner[57]=0;b.army[56]=b.army[57]=50;
  b.owner[59]=1;b.army[59]=70;b.structure[59]=1;
  auto p=std::make_unique<Policy>(0,1,"{}");b.refresh(*p);p->sources={56,57};
  p->occupationPrograms();
  assert(p->proposalCount>0);
  bool feasible=false;
  for(int j=0;j<p->proposalCount;j++){
   auto &q=p->proposals[j];
   if(q.kind==23&&q.goal==59&&q.count==3&&q.orders[0].s==56&&q.orders[0].t==57)feasible=true;
  }
  // Neither source can attack alone. The operation must first combine the
  // two real armies, and must stop proposing an attack when they are too small.
  assert(feasible);b.army[59]=200;b.refresh(*p);p->proposalCount=0;
  p->occupationPrograms();assert(p->proposalCount==0);
 }
 {
  Board b;std::fill(b.terrain,b.terrain+Board::S,1);b.terrain[0]=0;
  for(int c=55;c<66;c++)b.terrain[c]=0;
  b.owner[57]=0;b.army[57]=100;b.structure[57]=1;
  b.owner[59]=1;b.army[59]=60;
  auto p=std::make_unique<Policy>(0,1,"{}");auto action=p->act(&b.obs);
  // The sortie itself screens the city. A stationary half-garrison rule
  // cannot express this safe advance down a one-cell corridor.
  assert(p->reserve[57]>50);
  assert(action==arena_move(57,1,false));
 }
 {
  Board b;std::fill(b.terrain,b.terrain+Board::S,1);b.terrain[0]=0;
  for(int c=55;c<66;c++)b.terrain[c]=0;b.terrain[46]=0;
  b.owner[56]=b.owner[57]=0;b.army[56]=100;b.army[57]=50;b.structure[57]=1;
  b.owner[46]=1;b.army[46]=40;
  b.owner[59]=-2;b.army[59]=80;b.structure[59]=1;
  auto p=std::make_unique<Policy>(0,1,"{}");b.refresh(*p);
  assert(p->reserve[57]>1);
  p->campaignSearch(56,99,8);
  // Arrive with 149, grow to 150, then HALF sends exactly 75. Merely
  // subtracting a 40-troop reserve wrongly predicts a 110-troop sortie.
  assert(p->force[58]==75&&p->pathHalf[58]);
  assert(p->dist[59]>=INF);
 }
 {
  Board b;b.owner[60]=1;b.structure[60]=2;b.army[60]=130;
  b.owner[56]=b.owner[64]=0;b.army[56]=b.army[64]=80;
  auto p=std::make_unique<Policy>(0,1,"{}");b.refresh(*p);
  p->sources={56,64};p->convergingAssaultPrograms();
  assert(p->proposalCount>0&&p->proposals[0].kind==21);
  // Two approach routes sharing one donor are one army, not two waves.
  b.owner[64]=-1;b.army[64]=0;b.owner[58]=0;b.army[58]=30;
  b.refresh(*p);p->proposalCount=0;p->sources={56,58};p->convergingAssaultPrograms();
  assert(p->proposalCount==0);
 }
 {
  Board b;b.owner[60]=1;b.structure[60]=1;b.army[60]=10;
  auto p=std::make_unique<Policy>(0,1,"{}");b.refresh(*p);
  double hostile=p->assetUtility(60,70,1);
  b.owner[60]=-2;b.refresh(*p);double neutral=p->assetUtility(60,70,1);
  // The captured city's former owner is not itself a counterattacking army.
  // Retaining an enemy city changes both players' future production.
  assert(hostile>neutral*1.6);
  b.owner[60]=1;b.owner[61]=1;b.army[61]=100;b.refresh(*p);
  assert(p->assetUtility(60,70,1)<hostile*.5);
 }
 {
  Board b;b.owner[56]=0;b.army[56]=80;
  b.owner[58]=-2;b.structure[58]=1;b.army[58]=60;
  b.owner[60]=1;b.structure[60]=1;b.army[60]=5;
  auto p=std::make_unique<Policy>(0,1,"{}");b.refresh(*p);
  p->campaignSearch(56,79,16);
  assert(p->dist[60]>4&&p->force[60]>60);
  for(int c=60;c!=56;c=p->parent[c])assert(c>=0&&c!=58);
  // Economic pricing of enemy casualties must never invent actual troops.
  std::fill(b.terrain,b.terrain+Board::S,1);b.terrain[0]=0;
  for(int c=55;c<66;c++)b.terrain[c]=0;
  b.owner[58]=1;b.structure[58]=0;b.army[58]=40;b.refresh(*p);
  p->campaignSearch(56,79,16);assert(p->force[60]>20&&p->force[60]<35);
 }
 for(int seat=0;seat<2;seat++){
  Board b;b.obs.player=seat;b.owner[0]=seat;
  b.owner[60]=1-seat;b.structure[60]=2;b.army[60]=130;
  b.owner[59]=b.owner[61]=seat;b.army[59]=b.army[61]=80;
  auto p=std::make_unique<Policy>(seat,1,"{}");b.refresh(*p);
  p->coordinatedAssaultPrograms();
  // Neither column wins alone. Two attacks work against every visible
  // local response, including an earlier defender overrunning the first
  // source and a later defender withdrawing its general's garrison.
  assert(p->proposalCount>0&&p->proposals[0].kind==17);
  // A reinforcement from one step away is an executable defense, not a
  // hypothetical stationary garrison. It prevents this two-turn attack.
  b.owner[49]=1-seat;b.army[49]=60;b.refresh(*p);p->proposalCount=0;
  p->coordinatedAssaultPrograms();assert(p->proposalCount==0);
  // A donor two steps away can merge and reinforce over the two replies.
  b.owner[49]=1-seat;b.army[49]=1;b.owner[38]=1-seat;b.army[38]=80;
  b.refresh(*p);p->proposalCount=0;p->coordinatedAssaultPrograms();
  if(seat==1)assert(p->proposalCount==0);
 }
 {
  Board b;b.owner[1]=b.owner[11]=1;b.army[1]=80;b.army[11]=70;
  b.owner[22]=b.owner[23]=0;b.army[22]=b.army[23]=30;
  auto p=std::make_unique<Policy>(0,1,"{}");b.refresh(*p);
  assert(p->homeIncoming[1]<b.army[0]);
  assert(p->homeIncoming[2]>b.army[0]);
  assert(p->baselineDanger>0);
 }
 {
  Board b;b.owner[120]=b.owner[119]=b.owner[109]=1;b.structure[120]=2;
  b.army[120]=10;b.army[119]=40;b.army[109]=50;
  auto p=std::make_unique<Policy>(0,1,"{}");b.refresh(*p);
  // Two independent donors can reach the king in two orders. Assuming only
  // the single largest donor understates the defense of a distant attack.
  assert(p->kingRelief[1][3]>=88);
  assert(p->kingRelief[1][1]==0); // The defender acts after this seat.
 }
 {
  Board b;b.army[0]=200;b.owner[5]=1;b.army[5]=140;
  b.owner[11]=b.owner[12]=0;b.army[11]=b.army[12]=1;
  auto p=std::make_unique<Policy>(0,1,"{}");b.refresh(*p);
  bool half=false;assert(!p->canDepart(0,half));
  Policy::Program q;q.kind=14;q.root=0;q.goal=12;q.count=3;q.born=300;
  q.orders[0]={0,11,true};q.orders[1]={11,12,true};q.orders[2]={11,0,false};
  assert(p->extractionSafe(q));
  // The same sequence must be rejected when the attack can arrive before
  // the planned return, regardless of its final garrison being sufficient.
  b.owner[2]=1;b.army[2]=140;b.owner[5]=-1;b.army[5]=0;b.refresh(*p);
  assert(!p->extractionSafe(q));
 }
 {
  Board b;b.owner[59]=0;b.army[59]=200;b.owner[60]=1;b.army[60]=100;
  auto p=std::make_unique<Policy>(0,1,"{}");auto action=p->act(&b.obs);
  assert(action==arena_move(59,1,false));
  // The same exchange becomes unsound when a second visible army can take
  // the surviving column immediately; do not count the first kill alone.
  b.owner[61]=1;b.army[61]=200;
  auto q=std::make_unique<Policy>(0,1,"{}");action=q->act(&b.obs);
  assert(action!=arena_move(59,1,false));
 }
 {
  Board b;b.owner[58]=b.owner[59]=b.owner[60]=0;b.army[58]=20;b.army[59]=1;b.army[60]=10;
  auto p=std::make_unique<Policy>(0,1,"{}");b.refresh(*p);
  Policy::Program q;q.kind=11;q.born=300;q.root=q.goal=60;q.value=4;q.count=2;q.cursor=1;
  q.orders[0]={58,59,false};q.orders[1]={59,60,false};
  auto action=arena_move(58,1,false);p->rememberOperation(q,q.orders[0],action);
  // Online updates may skip a tick and omit native move receipts. The
  // operation is acknowledged from its own source and destination changes.
  b.obs.tick=302;b.army[58]=1;b.army[59]=20;b.refresh(*p);p->proposalCount=0;p->resumeOperation();
  assert(p->proposalCount==1&&p->proposals[0].orders[1].s==59);
  // An emergency elsewhere does not discard the suspended collection tree.
  b.obs.tick=303;b.refresh(*p);p->proposalCount=0;p->resumeOperation();
  assert(p->proposalCount==1&&p->proposals[0].orders[1].s==59);
  // Losing the remaining donor makes the plan infeasible, so it is cancelled.
  b.obs.tick=304;b.owner[59]=1;b.army[59]=80;b.refresh(*p);p->proposalCount=0;p->resumeOperation();
  assert(p->proposalCount==0);
 }
 {
  // A second independent operation must not erase an interrupted column.
  Board b;b.owner[55]=b.owner[56]=b.owner[57]=0;b.army[55]=60;b.army[56]=b.army[57]=1;
  b.owner[77]=b.owner[78]=b.owner[79]=0;b.army[77]=30;b.army[78]=b.army[79]=1;
  auto p=std::make_unique<Policy>(0,1,"{}");b.refresh(*p);
  Policy::Program first;first.kind=1;first.born=300;first.root=55;first.goal=58;
  first.value=4;first.count=3;first.cursor=1;
  first.orders[0]={55,56,false};first.orders[1]={56,57,false};first.orders[2]={57,58,false};
  p->rememberOperation(first,first.orders[0],arena_move(55,1,false));
  b.obs.tick=301;b.army[55]=1;b.army[56]=60;b.refresh(*p);p->proposalCount=0;p->resumeOperation();
  assert(p->proposalCount==1);
  Policy::Program second;second.kind=2;second.born=301;second.root=77;second.goal=79;
  second.value=4;second.count=2;second.cursor=1;second.orders[0]={77,78,false};second.orders[1]={78,79,false};
  p->rememberOperation(second,second.orders[0],arena_move(77,1,false));
  b.obs.tick=302;b.army[77]=1;b.army[78]=30;b.refresh(*p);p->proposalCount=0;p->resumeOperation();
  bool a=false,z=false;for(int i=0;i<p->proposalCount;i++){
   auto &q=p->proposals[i];a|=q.orders[q.cursor].s==56;z|=q.orders[q.cursor].s==78;
  }
  assert(a&&z);
  // An actual emergency withdrawal consumes the first task's army. That
  // task must be cancelled, while the independent supply task survives.
  Policy::Program emergency;emergency.kind=3;emergency.count=1;
  p->rememberOperation(emergency,{56,45,false},arena_move(56,0,false));
  b.obs.tick=303;b.army[56]=1;b.owner[45]=0;b.army[45]=59;
  b.refresh(*p);p->proposalCount=0;p->resumeOperation();
  assert(p->proposalCount==1&&p->proposals[0].orders[p->proposals[0].cursor].s==78);
 }
 {
  Board b;b.owner[59]=b.owner[60]=0;b.army[59]=25;b.army[60]=25;
  auto p=std::make_unique<Policy>(0,1,"{}");b.refresh(*p);
  Policy::Program ongoing;ongoing.kind=11;ongoing.born=299;ongoing.value=4;
  ongoing.count=1;ongoing.orders[0]={59,60,false};
  Policy::Program fresh=ongoing;fresh.born=300;fresh.value=4.5;
  p->offer(ongoing);p->offer(fresh);
  // Deduplication and final selection must use the same commitment utility.
  // Otherwise the identical first action silently erases the longer plan.
  assert(p->proposalCount==1&&p->proposals[0].born==299);
 }
 {
  Board b;b.owner[60]=b.owner[59]=0;b.structure[60]=1;b.army[60]=100;b.army[59]=1;
  b.owner[62]=1;b.army[62]=45;
  auto p=std::make_unique<Policy>(0,1,"{}");b.refresh(*p);
  // A two-step observed counterattack must affect departure, even though no
  // enemy is adjacent to the city yet. Half departure can retain production.
  assert(!p->permitted({60,59,false}));
  assert(p->permitted({60,59,true}));
  p->scheduleTree(59,1);Policy::Program q;p->extractSchedule(59,1,q);
  assert(q.count==1&&q.orders[0].s==60&&q.orders[0].half);
  assert(p->schedule[59][1]==51);
 }
 {
  Board b;b.structure[60]=1;b.owner[60]=1;b.army[60]=4;
  auto p=std::make_unique<Policy>(0,1,"{}");b.refresh(*p);
  double original=p->assetUtility(60,10,1);
  for(int cycle=0;cycle<3;cycle++){
   b.obs.tick+=2;b.owner[60]=0;b.army[60]=10;b.refresh(*p);
   b.obs.tick+=2;b.owner[60]=1;b.army[60]=4;b.refresh(*p);
  }
  assert(p->assetUtility(60,10,1)<original*.6);
  // A new force capable of retaining the asset restores the investment.
  assert(p->assetUtility(60,80,1)>original*.9);
 }
 {
  Board b;b.owner[56]=0;b.army[56]=35;b.owner[60]=1;b.army[60]=180;
  auto p=std::make_unique<Policy>(0,1,"{}");b.refresh(*p);
  p->taskRoutes(56,34,true);
  assert(p->opSteps[64]<INF);
  // The information route deliberately detours around a visible main army.
  int steps=0;for(int c=64;c!=56;c=p->opParent[c]){
   assert(c>=0&&c!=60&&++steps<40);
  }
  assert(steps>8);
 }
}
