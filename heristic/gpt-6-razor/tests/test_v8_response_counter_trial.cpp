#include "../experiments/7e426fa94ce8ec8c0f84a3a61d01af9d141d9c37915f00e15a703bc694f948e7/policy.cpp"
#include <cassert>
#include <iostream>
#include <memory>
struct Board {
 static constexpr int N=289;
 uint8_t terrain[N],structure[N]{},visible[N],alive[2]={1,1};
 int16_t owner[N];int64_t army[N]{},total[2]={240,350},land[2]={20,34},cities[2]{};
 ArenaObservation obs{17,17,2,1,300,0,terrain,structure,owner,army,visible,alive,total,land,cities,nullptr};
 Board(){
  std::fill(terrain,terrain+N,1);std::fill(visible,visible+N,1);std::fill(owner,owner+N,-1);
  for(int c=0;c<=16;c++){terrain[c]=0;owner[c]=1;army[c]=1;}
  terrain[17]=0;owner[17]=1;army[17]=1;
  for(int c=34;c<=50;c++){terrain[c]=0;owner[c]=1;army[c]=1;}
  structure[0]=structure[16]=2;army[0]=10;owner[16]=0;army[16]=200;army[46]=300;
 }
 void setup(Policy &p){p.planningStart=std::chrono::steady_clock::now();p.update(&obs);p.assetHistory();p.pressureMap();p.buildAttacks();p.baselineDanger=p.dangerAfter(nullptr,p.homeIncoming);p.prepareResponseBudget();}
};
int main(){
 Board b;auto p=std::make_unique<Policy>(1,1,"{}");b.setup(*p);
 assert(p->responseActive&&p->responseEnemy==16&&p->responseBaseline.count==14);
 Policy::Order outward{46,47,false};Policy::Program after;
 assert(!p->responsePlan(&outward,after));
 Policy::Order wait;assert(p->responsePlan(&wait,after));
 p->responseLongRecall=true;p->proposalCount=1;
 p->proposals[0]=p->oneStep({10,46,47,1,false},1);p->proposalScore[0]=10;
 Policy::Program chosen;assert(p->selectResponseBudget(chosen)&&chosen.count==0);
 // The same distant attacker actually advances. One more idle command now
 // loses the last response window, so dispatch the real collection instead.
 b.owner[15]=0;b.army[15]=199;b.army[16]=1;b.obs.tick=301;b.setup(*p);
 assert(p->responseActive&&p->responseEnemy==15&&p->responseBaseline.count==14);
 assert(!p->responsePlan(&wait,after));
 p->responseLongRecall=true;p->proposalCount=1;p->proposals[0]=p->oneStep({10,46,47,1,false},1);p->proposalScore[0]=10;
 assert(p->selectResponseBudget(chosen)&&chosen.count==14);
 assert(chosen.orders[0].s==46&&chosen.orders[0].t==45);
 // Seat order: a donor removed by the earlier enemy cannot still be sent.
 b.army[14]=80;b.setup(*p);
 Policy::Program cut;cut.count=1;cut.orders[0]={14,13,false};
 assert(!p->responseOutcome(cut,p->responseRoutes[0]));
 // Model limits are explicit; a second comparable column restores the
 // original controller rather than pretending it belongs to the first raid.
 b.owner[50]=0;b.army[50]=150;b.setup(*p);assert(!p->responseActive);
 {
  Board race;race.army[46]=1;race.army[16]=100;race.army[14]=40;
  for(int c:{31,32,33}){race.terrain[c]=0;race.owner[c]=1;race.army[c]=1;}
  auto agent=std::make_unique<Policy>(1,2,"{}");race.setup(*agent);
  Policy::Order flank{14,31,false};Policy::Program answer;
  // A weak counter can kill the empty general after a full launch, but
  // cannot defeat the garrison preserved by a lethal half launch.
  assert(agent->responseCount>=2);
  assert(!agent->responseCounter(flank,answer));
  race.army[14]=80;race.setup(*agent);
  assert(agent->responseCounter(flank,answer));
  assert(answer.goal==16&&answer.orders[0].s==14&&answer.orders[0].t==31);
 }
 std::cout<<"wait-preserves-position; advancing-threat-recalls; cut-donor-rejected; second-column-fallback\n";
}
