#include "../experiments/8eefe0dd592a9215cb8972eed710bb3ece574fb9adff88cd7d7fde19defaa66e/policy.cpp"
#include <cassert>
#include <iostream>
#include <memory>
struct Board {
 static constexpr int N=169;
 uint8_t terrain[N],structure[N]{},visible[N],alive[2]={1,1};
 int16_t owner[N];int64_t army[N]{},total[2]={52,60},land[2]={4,5},cities[2]{};
 ArenaObservation obs{13,13,2,1,200,0,terrain,structure,owner,army,visible,alive,total,land,cities,nullptr};
 Board(){std::fill(visible,visible+N,1);std::fill(terrain,terrain+N,1);std::fill(owner,owner+N,-1);
  for(int c=0;c<=4;c++){terrain[c]=0;owner[c]=0;army[c]=1;}
  for(int c:{0,1,13,26}){terrain[c]=0;owner[c]=1;army[c]=1;}
  army[0]=35;structure[0]=2;army[4]=40;army[2]=10;army[13]=4;army[26]=8;
 }
 void setup(Policy &p){p.planningStart=std::chrono::steady_clock::now();p.update(&obs);p.assetHistory();p.pressureMap();p.buildAttacks();p.baselineDanger=p.dangerAfter(nullptr,p.homeIncoming);}
};
int main(){
 Board b;auto p=std::make_unique<Policy>(1,3,"{}");b.setup(*p);
 assert(p->baselineDanger>0&&p->defend().source<0);
 Policy::Program weak;weak.count=1;weak.orders[0]={13,0,false};assert(!p->rescueHolds(weak));
 p->proposalCount=0;p->rescuePrograms();assert(p->proposalCount==1);
 auto q=p->proposals[0];assert(q.rescueModel&&q.count==2&&p->rescueHolds(q));
 assert(q.orders[0].s==26&&q.orders[0].t==13&&q.orders[1].t==0);
 // An already moving second reinforcement can be more valuable than the
 // minimum force that survives immediate raids. Price completed collections
 // against the current readiness target, not the first feasible budget.
 Board ample;ample.terrain[39]=ample.terrain[160]=0;ample.owner[39]=ample.owner[160]=1;
 ample.army[39]=40;ample.army[160]=100;ample.total[1]=188;
 auto margin=std::make_unique<Policy>(1,3,"{}");ample.setup(*margin);
 margin->proposalCount=0;margin->rescuePrograms();assert(margin->proposalCount==1);
 assert(margin->proposals[0].count==3&&margin->proposals[0].orders[0].s==39);
 // A queued donor lost to an earlier seat cannot still reinforce home.
 Board c;c.army[0]=3;c.army[1]=5;c.army[2]=10;c.army[3]=c.army[4]=1;c.setup(*p);
 Policy::AttackRoute raid;raid.owner=0;raid.length=2;raid.cell[0]=2;raid.cell[1]=1;raid.cell[2]=0;
 Policy::Program donor;donor.count=1;donor.orders[0]={1,0,false};
 assert(!p->rescueRoute(donor,&raid,nullptr,0));
 // Swap execution order while preserving the same visible physical position.
 c.obs.player=0;for(int i=0;i<c.N;i++)if(c.owner[i]>=0)c.owner[i]=1-c.owner[i];
 auto first=std::make_unique<Policy>(0,3,"{}");c.setup(*first);raid.owner=1;
 assert(first->rescueRoute(donor,&raid,nullptr,0));
 // Once a march is absorbed, a pre-existing later guard is not resurrected
 // into the same column. It remains a separate attack scenario instead.
 b.army[0]=3;b.owner[3]=1;b.army[3]=50;b.army[2]=100;b.setup(*p);
 raid.owner=0;raid.length=4;for(int i=0;i<=4;i++)raid.cell[i]=4-i;
 assert(p->rescueRoute(weak,&raid,nullptr,0));
 assert(!p->rescueHolds(weak));
 // QueueT game4 at189 contains this local geometry: a diagonal enemy
 // reaches either side of home in one step. The apparent shortest donor
 // route can be cut; moving the same donor around the back holds both.
 Board d;std::fill(d.terrain,d.terrain+d.N,1);std::fill(d.owner,d.owner+d.N,-1);
 std::fill(d.army,d.army+d.N,0);std::fill(d.structure,d.structure+d.N,0);
 for(int c:{28,42,16,15,29,41}){d.terrain[c]=0;d.owner[c]=0;d.army[c]=1;}
 d.owner[42]=1;d.army[42]=41;d.army[28]=35;d.army[16]=3;
 d.structure[28]=2;d.obs.general_pos=28;d.obs.player=0;d.obs.tick=189;
 auto diamond=std::make_unique<Policy>(0,3,"{}");d.setup(*diamond);
 Policy::Program exposed;exposed.count=2;exposed.orders[0]={16,29,false};exposed.orders[1]={29,28,false};
 assert(!diamond->rescueHolds(exposed));
 Policy::Program covered;covered.count=2;covered.orders[0]={16,15,false};covered.orders[1]={15,28,false};
 assert(diamond->rescueHolds(covered));
 diamond->proposalCount=0;diamond->rescuePrograms();assert(diamond->proposalCount==1);
 assert(diamond->proposals[0].orders[0].s==16&&diamond->proposals[0].orders[0].t==15);
 std::cout<<"merged-threat-needs-complete-rescue; partial-transfer-rejected; source-capture-respects-seat-order; stopped-column-not-resurrected; alternate-approach-uses-covered-donor\n";
}
