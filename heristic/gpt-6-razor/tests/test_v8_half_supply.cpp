#include "../experiments/ed3370d06be60a25bc96e8040dc57bd04f491915fc16a0a9c1086810f6c95194/policy.cpp"
#include <cassert>
#include <memory>
int main(){
 constexpr int N=121;
 uint8_t terrain[N],structure[N]{},visible[N],alive[2]={1,1};
 int16_t owner[N];int64_t army[N]{},total[2]={111,100},land[2]={3,20},cities[2]={0,-1};
 std::fill(terrain,terrain+N,1);std::fill(visible,visible+N,1);std::fill(owner,owner+N,-1);
 for(int c:{0,1,11}){terrain[c]=0;owner[c]=0;army[c]=1;}
 structure[0]=2;army[0]=80;army[11]=30;
 ArenaObservation obs{11,11,2,0,301,0,terrain,structure,owner,army,visible,alive,total,land,cities,nullptr};
 auto p=std::make_unique<Policy>(0,1,"{}");p->update(&obs);p->reserve[0]=50;
 p->scheduleTree(1,2);assert(p->schedule[1][2]==56);
 Policy::Program q;q.root=1;q.goal=1;p->extractSchedule(1,2,q);
 assert(q.count==2&&q.orders[0].s==11&&q.orders[0].t==0&&!q.orders[0].half);
 assert(q.orders[1].s==0&&q.orders[1].t==1&&q.orders[1].half);
 assert(p->collectionArrival(q,1)==56);
 int troops[N]{};for(int c:{0,1,11})troops[c]=p->ready(c);
 int before=troops[0]+troops[1]+troops[11];
 for(int j=0;j<q.count;j++){
  if(j)for(int c:{0,1,11})troops[c]+=p->grows(c,p->actionTurn(j+1));
  auto x=q.orders[j];int sent=x.half?troops[x.s]/2:troops[x.s]-1;
  troops[x.s]-=sent;troops[x.t]+=sent;
  assert(troops[x.s]>=p->reserve[x.s]);
 }
 assert(troops[0]==55&&troops[1]==56&&troops[11]==1);
 assert(troops[0]+troops[1]+troops[11]==before);
 // A higher required garrison invalidates the half after the same merge.
 p->reserve[0]=61;p->scheduleTree(1,2);assert(p->schedule[1][2]<0);
}
