#include "../experiments/bc91301db23eda7ec099d8a8ac14bf1a4e5086684bd3fec4d310bd488cd7ef3c/policy.cpp"
#include <cassert>
#include <memory>

int main() {
 constexpr int N=49;
 uint8_t terrain[N]{},structure[N]{},visible[N],alive[2]={1,1};
 int16_t owner[N];int64_t army[N],total[2]={150,150},land[2]={20,20},cities[2]{};
 std::fill(visible,visible+N,1);std::fill(owner,owner+N,0);std::fill(army,army+N,1);
 structure[0]=2;army[0]=30;
 ArenaObservation obs{7,7,2,0,185,0,terrain,structure,owner,army,visible,alive,total,land,cities,nullptr};
 auto p=std::make_unique<Policy>(0,1,"{}");p->update(&obs);
 p->tacticalFocus=24;p->tacticalEnemy=1;p->tacticalHome=p->tacticalKing=-1;
 p->tacticalCount=0;
 for(int c=0;c<N;c++)if(std::abs(c/7-3)+std::abs(c%7-3)<=2)
  p->tacticalCells[p->tacticalCount++]=c;
 for(int i=0;i<p->tacticalCount;i++)for(int k=0;k<4;k++){
  p->tacticalNeighbors[i][k]=-1;
  for(int j=0;j<p->tacticalCount;j++)if(p->tacticalCells[j]==p->nb[p->tacticalCells[i]][k])
   p->tacticalNeighbors[i][k]=j;
 }
 for(int side=0;side<2;side++){
  Policy::TacticalPosition position{};int root=-1;
  for(int i=0;i<p->tacticalCount;i++){
   position.owner[i]=side;position.army[i]=2;
   p->tacticalProjection[side][i]=p->tacticalCells[i]==24?.5:0.;
   if(p->tacticalCells[i]==24){root=i;position.army[i]=100;}
  }
  // Every full retreat has worse immediate projection than a split or a
  // small unrelated transfer. Its two-round consequence must still be seen.
  for(int limit:{6,9}){
   Policy::TacticalOrder moves[10];moves[limit].s=12345;
   p->planningStart=std::chrono::steady_clock::now();p->tacticalWork=0;p->tacticalAborted=false;
   int count=p->tacticalOrders(position,side,moves,limit);
   assert(!p->tacticalAborted&&count<=limit&&moves[0].s==-1&&moves[limit].s==12345);
   for(int direction=0;direction<4;direction++){
    int target=p->tacticalNeighbors[root][direction];assert(target>=0);
    bool found=false;for(int i=1;i<count;i++)
     found|=moves[i].s==root&&moves[i].t==target&&!moves[i].half;
    assert(found);
   }
  }
 }
}
