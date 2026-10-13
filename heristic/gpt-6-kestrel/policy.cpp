#include "arena_policy.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

namespace {
constexpr int MAXN=4096, INF=1000000;
struct Candidate { double score=-1e30; int source=-1, target=-1, dir=0; bool half=false; };
struct Policy {
 int p,W=0,H=0,N=0,g=-1,tick=0,oldTarget=-1,oldSource=-1,lastTarget=-1;
 uint64_t rng;
 int ter[MAXN],st[MAXN],own[MAXN],seen[MAXN],a[MAXN],lastOwn[MAXN];
 int nb[MAXN][4],gd[MAXN],fd[MAXN],ed[MAXN],unknown[MAXN],reserve[MAXN],targetDist[MAXN];
 int dist[MAXN],force[MAXN],pathLoss[MAXN],first[MAXN],parent[MAXN],queue[MAXN];
 double threat[MAXN];
 int threatETA[MAXN], attackCost[MAXN], nextAttackCost[MAXN],attackGuard[MAXN];
 int homeCells[MAXN],homeCount=0;
 std::uint64_t searchTargets[MAXN/64];
 int rememberedReserve=1, rememberedReserveTick=-1000;
 int closeAttackers[20], closeAttackerCount=0;
 int closeOwnerArmy[8], closeOwnerETA[8], closeOwnerSource[8];
 int coalitionNeed=0, coalitionETA=INF, coalitionDeadline=INF, coalitionSource=-1, coalitionCount=0;
 int enemyGeneral[8], oldTotals[8]{}; double growth[8]{};
 int armyCache[MAXN],fullMoveCache[MAXN],halfMoveCache[MAXN],defenseCache[MAXN],routeDelta[MAXN];
 bool geometryReady=false;
 const ArenaObservation *o=nullptr;
 Policy(int player,uint64_t seed):p(player),rng(seed) {
  std::fill(ter,ter+MAXN,255);std::fill(st,st+MAXN,255);std::fill(own,own+MAXN,-3);
  std::fill(a,a+MAXN,0);std::fill(seen,seen+MAXN,-10000);std::fill(lastOwn,lastOwn+MAXN,-3);
  std::fill(enemyGeneral,enemyGeneral+8,-1);
 }
 uint64_t random(){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;return rng;}
 bool mine(int i)const{return i>=0 && o->owner[i]==p;}
 int army(int i)const{return armyCache[i];}
 int moveArmy(int i,bool half=false)const {return half?halfMoveCache[i]:fullMoveCache[i];}
 bool canMoveNow(int i)const {return o->players==2?moveArmy(i)>0:army(i)>1;}
 void bfs(int root,int *out,bool friends=false,bool noCities=false){
  std::fill(out,out+N,INF);if(root<0)return;int h=0,z=0;queue[z++]=root;out[root]=0;
  while(h<z){int u=queue[h++];for(int k=0;k<4;k++){int v=nb[u][k];if(v<0||ter[v]==1||out[v]!=INF||(friends&&!mine(v))||(noCities&&st[v]==1&&!mine(v)))continue;out[v]=out[u]+1;queue[z++]=v;}}
 }
 // Cache exactly the cells visited by the original twelve-layer defense DP.
 void cacheHomeGeometry(){
  homeCount=0;
  for(int c=0;c<N;c++)if(gd[c]<=12)homeCells[homeCount++]=c;
 }
 void update(const ArenaObservation *obs){
  o=obs;tick=int(o->tick);g=o->general_pos;
  if(W==0){W=o->width;H=o->height;N=W*H;for(int i=0;i<N;i++){int x=i%W,y=i/W;nb[i][0]=y?i-W:-1;nb[i][1]=x+1<W?i+1:-1;nb[i][2]=y+1<H?i+W:-1;nb[i][3]=x?i-1:-1;}}
  bool mapChanged=!geometryReady;
  for(int i=0;i<N;i++)armyCache[i]=int(std::min<int64_t>(10000000,std::max<int64_t>(0,o->army[i])));
  for(int i=0;i<N;i++){
   if(o->visible[i]&&ter[i]!=o->terrain[i])mapChanged=true;
   lastOwn[i]=own[i];if(o->visible[i]){ter[i]=o->terrain[i];st[i]=o->structure[i];own[i]=o->owner[i];a[i]=army(i);seen[i]=tick;if(st[i]==2&&own[i]>=0&&own[i]!=p)enemyGeneral[own[i]]=i;}
   else if(own[i]==p){own[i]=-3;a[i]=0;}
   else if(own[i]>=0&&!o->alive[own[i]]){own[i]=-3;a[i]=std::max(1,a[i]/2);if(st[i]==2)st[i]=1;}
   reserve[i]=1;threat[i]=0;threatETA[i]=INF;
  }
  // Observation and remembered terrain are fixed for this decision. Keep the
  // original growth-before-action expressions, evaluated once per cell.
  for(int i=0;i<N;i++){
   int n=armyCache[i];
   if((tick+1)%2==0&&st[i]>0&&st[i]<3)n++;
   else if((tick+1)%50==0&&st[i]==0)n++;
   fullMoveCache[i]=std::max(0,n-1);halfMoveCache[i]=n/2;
   if(mine(i))defenseCache[i]=0;
   else if(own[i]<0)defenseCache[i]=std::max(0,a[i]);
   else{
    int add=(st[i]>0&&st[i]<3)?std::max(0,tick-seen[i])/2:0;
    defenseCache[i]=a[i]+std::min(add,200)+int(o->visible[i]&&(((tick+1)%2==0&&st[i]>0&&st[i]<3)||((tick+1)%50==0&&st[i]==0)));
    if(o->players>=7&&!o->visible[i]){
     // A last-seen field stack is mobile. Keep the observation in memory,
     // but let its planning estimate fade; the actual first step is visible.
     if(st[i]==0){
      int age=std::max(0,tick-seen[i]);
      int floor=(a[i]>0?1:0)+std::max(0,tick/50-seen[i]/50);
      int mobile=int(int64_t(a[i])*std::max(0,24-age)/24);
      defenseCache[i]=std::max(floor,mobile);
     }
     // A remembered single garrison cannot exceed that owner's entire
     // public current army, apart from its next one-troop growth event.
     int publicCap=int(std::min<int64_t>(10000000,std::max<int64_t>(0,o->total_army[own[i]])))+1;
     defenseCache[i]=std::min(defenseCache[i],publicCap);
    }
   }
  }
  for(int e=0;e<o->players;e++){if(!o->alive[e])enemyGeneral[e]=-1;oldTotals[e]=int(o->total_army[e]);}
  if(mapChanged){bfs(g,gd,false,false);cacheHomeGeometry();}
  std::fill(ed,ed+N,INF);int h=0,z=0;
  for(int i=0;i<N;i++)if(o->visible[i]&&own[i]>=0&&own[i]!=p){ed[i]=0;queue[z++]=i;}
  while(h<z){int u=queue[h++];for(int k=0;k<4;k++){int v=nb[u][k];if(v>=0&&ter[v]!=1&&ed[v]==INF){ed[v]=ed[u]+1;queue[z++]=v;}}}
  if(mapChanged)for(int i=0;i<N;i++){
   unknown[i]=0;int x=i%W,y=i/W;for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){int xx=x+dx,yy=y+dy;if(xx>=0&&xx<W&&yy>=0&&yy<H&&ter[yy*W+xx]==255)unknown[i]++;}
  }
  geometryReady=true;
  // Minimum troop loss of an enemy route to our general, up to 12 moves.
  // Our current blocking armies and neutral city garrisons really do absorb
  // attackers. Third-party troops are not trusted as permanent protection.
  // Costs are positive, so considering walks cannot create free army cycles.
  std::fill(closeOwnerArmy,closeOwnerArmy+8,0);
  std::fill(closeOwnerETA,closeOwnerETA+8,INF);
  std::fill(closeOwnerSource,closeOwnerSource+8,-1);
  std::fill(attackCost,attackCost+N,INF);
  std::fill(nextAttackCost,nextAttackCost+N,INF);attackCost[g]=0;
  for(int j=0;j<homeCount;j++){
   int c=homeCells[j];attackGuard[c]=0;
   if(c!=g){
    if(mine(c))attackGuard[c]=army(c);
    else if(o->visible[c]&&own[c]==-2)attackGuard[c]=std::max(0,a[c]);
   }
  }
  for(int d=1;d<=12;d++){
   for(int j=0;j<homeCount;j++){
    int u=homeCells[j];if(u==g)continue;
    int cost=INF;
    for(int k=0;k<4;k++){
     int v=nb[u][k];if(v<0||attackCost[v]>=INF)continue;
     cost=std::min(cost,attackCost[v]+attackGuard[v]+1);
    }
    nextAttackCost[u]=cost;
    if(o->visible[u]&&own[u]>=0&&own[u]!=p&&a[u]>1&&cost<INF){
     int start=moveArmy(u)+1;
     int kingGrowth=(tick+d)/2-tick/2;
     int required=std::max(0,start-cost-kingGrowth);
     // Different owners can each act during one tick. Within an owner use
     // only its strongest nearby stack; never sum all of its candidates.
     if(o->players>2&&d<=2){
      int e=own[u],arriving=std::max(0,start-cost);
      if(arriving>closeOwnerArmy[e]||(arriving==closeOwnerArmy[e]&&d<closeOwnerETA[e])){
       closeOwnerArmy[e]=arriving;closeOwnerETA[e]=d;closeOwnerSource[e]=u;
      }
     }
     
     if(threatETA[u]==INF||required>threat[u]||(required==threat[u]&&d<threatETA[u])){threat[u]=required;threatETA[u]=d;}
    }
   }
   for(int j=0;j<homeCount;j++){int c=homeCells[j];attackCost[c]=nextAttackCost[c];}
  }
  int need=1;closeAttackerCount=0;
  for(int i=0;i<N;i++)if(threatETA[i]<INF){
   if(threatETA[i]<=2&&closeAttackerCount<20)closeAttackers[closeAttackerCount++]=i;
   // A single spare troop covers a close attack; distant blocked stacks no
   // longer immobilize the general solely because they are visible.
   int req=int(threat[i])+(threatETA[i]<=3?1:0);
   need=std::max(need,req);
  }
  coalitionNeed=0;coalitionETA=INF;coalitionDeadline=INF;coalitionSource=-1;coalitionCount=0;
  int combinedArrival=0,lastArrival=0;
  for(int e=0;e<o->players;e++)if(e!=p&&o->alive[e]&&closeOwnerArmy[e]>0){
   coalitionCount++;combinedArrival+=closeOwnerArmy[e];
   lastArrival=std::max(lastArrival,closeOwnerETA[e]);
   if(closeOwnerETA[e]<coalitionETA){coalitionETA=closeOwnerETA[e];coalitionSource=closeOwnerSource[e];}
  }
  if(coalitionCount>=2){
   // General growth occurs once, even when multiple opponents attack.
   coalitionNeed=std::max(0,combinedArrival-((tick+lastArrival)/2-tick/2));
   coalitionDeadline=coalitionETA;
   int beforeOurTurn=0;
   for(int e=0;e<p;e++)if(closeOwnerETA[e]==coalitionETA)beforeOurTurn+=closeOwnerArmy[e];
   // A harmless early-seat hit still leaves time for our same-tick relief;
   // shorten the deadline only when that earlier prefix is itself lethal.
   if(beforeOurTurn>army(g)+((tick+coalitionETA)/2-tick/2))coalitionDeadline--;
   need=std::max(need,coalitionNeed+1);
  }
  // Brief loss of vision must not authorize emptying home into a known raid.
  // Retain the recent peak for 16 ticks, fade it out over the next 12, then
  // expire it. The peak itself is not overwritten by the faded value.
  if(need>=rememberedReserve||tick-rememberedReserveTick>28){
   rememberedReserve=need;rememberedReserveTick=tick;
  }
  int age=tick-rememberedReserveTick;
  int retained=rememberedReserve;
  if(age>16)retained=rememberedReserve*std::max(0,28-age)/12;
  int garrison=tick>100?std::min(70,std::max(3+tick/60,4+int(o->total_army[p]/25))):1;
  reserve[g]=std::max(garrison,std::max(need,retained));
  for(int i=0;i<N;i++){
   if(mine(i)){
    routeDelta[i]=std::max(0,army(i)-reserve[i]+1);
    if(i==g&&reserve[i]>1)routeDelta[i]=0;
   }else routeDelta[i]=-defenseCache[i]-(ter[i]==255?1:0);
  }
 }
 int rememberedDefense(int i)const {return defenseCache[i];}
 bool canDepart(int s,bool &half)const{
  half=false;int n=o->players==2?moveArmy(s)+1:army(s);if(n<2)return false;
  if(reserve[s]>1){if(n-n/2>=reserve[s])half=true;else return false;}
  return true;
 }
 // Fast bounded breadth-first campaigns, keeping the strongest shortest route.
 void search(int s,int initial,int depth){
  std::fill(dist,dist+N,INF);std::fill(force,force+N,-INF);
  std::fill(searchTargets,searchTargets+(N+63)/64,0);int h=0,z=0;
  queue[z++]=s;dist[s]=0;force[s]=initial;pathLoss[s]=0;first[s]=-1;parent[s]=-1;
  while(h<z){int u=queue[h++];if(dist[u]>=depth)continue;
   for(int k=0;k<4;k++){int v=nb[u][k];if(v<0||ter[v]==1||v==s)continue;int d=dist[u]+1;if(dist[v]<d)continue;
    int f=force[u];if(u!=s)f-=1;
    f+=routeDelta[v];
    if(f<1)continue;
    if(dist[v]==INF){
     dist[v]=d;queue[z++]=v;
     if(!mine(v))searchTargets[v>>6]|=std::uint64_t(1)<<(v&63);
    }
    int spent=pathLoss[u]+int(u!=s)+(mine(v)?0:rememberedDefense(v)+int(ter[v]==255));
    if(f>force[v]||(f==force[v]&&spent<pathLoss[v])){force[v]=f;pathLoss[v]=spent;first[v]=(u==s?k:first[u]);parent[v]=u;}
   }
  }
 }
 double risk(int source,int dest,int moving)const {
  double cost=0;int remain=moveArmy(source)+1-moving;
  // A screen next to the general cannot be withdrawn while a hostile stack
  // can cross it and kill home before another friendly action can arrive.
  if(source!=g&&gd[source]==1&&dest!=g){
   int home=army(g)+((tick+2)/2-tick/2);
   for(int k=0;k<4;k++){
    int e=nb[source][k];if(e<0||!o->visible[e]||own[e]<0||own[e]==p)continue;
    if(e==dest&&moving>rememberedDefense(e))continue;
    if(moveArmy(e)-remain-1>home)return 100000;
   }
  }
  if(source==g){
   int garrison=tick>100?std::min(70,std::max(3+tick/60,4+int(o->total_army[p]/25))):1;
   if(remain<garrison)return 100000;
   for(int j=0;j<closeAttackerCount;j++){
    int e=closeAttackers[j];if(threat[e]<=remain)continue;
    if(e==dest&&moving>rememberedDefense(e))continue;
    return 100000;
   }
  }
  for(int k=0;k<4;k++){int e=nb[source][k];if(e>=0&&e!=dest&&o->visible[e]&&own[e]>=0&&own[e]!=p&&a[e]>remain+1){
    if(source==g)return 100000;
    if(st[source]==1)cost+=5;
  }}
  return cost;
 }
 // An earlier-seat victim may reinforce its general before our preselected
 // action executes. One owner has only one action, so take its largest legal
 // adjacent transfer instead of summing all possible reinforcements.
 int captureDefense(int dest)const {
  int victim=own[dest],reinforcement=0;
  if(victim>=0&&victim<p){
   for(int k=0;k<4;k++){
    int source=nb[dest][k];
    if(source>=0&&o->visible[source]&&own[source]==victim)
     reinforcement=std::max(reinforcement,moveArmy(source));
   }
  }
  return rememberedDefense(dest)+reinforcement;
 }
 // Only surviving opponents threaten home after a one-step elimination.
 // Include the victim's inherited troops as local screens, without imposing
 // the normal peacetime garrison on an immediately successful capture.
 bool captureSafe(int source,int dest,int moving)const {
  int victim=own[dest];
  auto grows=[&](int c,int at){return int((at%2==0&&st[c]>0&&st[c]<3)||(at%50==0&&st[c]==0));};
  auto ready=[&](int c){return army(c)+(own[c]>=0?grows(c,tick+1):0);};
  auto guard=[&](int c){
   if(c==dest)return moving-captureDefense(dest);
   if(c==source)return ready(c)-moving;
   if(mine(c))return ready(c);
   // A victim that acts earlier could evacuate this stack before capture;
   // only its compulsory one-troop remainder is a reliable inherited screen.
   if(o->visible[c]&&own[c]==victim)return victim<p?1:std::max(1,ready(c)/2);
   if(o->visible[c]&&own[c]==-2)return army(c);
   return 0;
  };
  int arriving[8][3]{};
  for(int e=0;e<N;e++)if(o->visible[e]&&own[e]>=0&&own[e]!=p&&own[e]!=victim&&o->alive[own[e]]&&gd[e]<=2){
   int enemy=own[e],sent=ready(e)-1;if(sent<=0)continue;
   for(int k=0;k<4;k++){
    int v=nb[e][k];if(v<0||ter[v]==1)continue;
    if(v==g){
     arriving[enemy][1]=std::max(arriving[enemy][1],sent);
     arriving[enemy][2]=std::max(arriving[enemy][2],sent+grows(e,tick+2));
    }else if(gd[v]==1){
     int next=sent-guard(v);
     if(o->visible[v]&&own[v]==enemy)next+=ready(v);
     if(next<=0)continue;
     int attack=next+grows(v,tick+2)-1;
     arriving[enemy][2]=std::max(arriving[enemy][2],attack);
    }
   }
  }
  int home=ready(g)-(source==g?moving:0),byOwner[8]{};
  for(int d=1;d<=2;d++){
   int incoming=0;
   for(int e=0;e<o->players;e++)if(e!=p&&e!=victim&&o->alive[e]){
    byOwner[e]=std::max(byOwner[e],arriving[e][d]);incoming+=byOwner[e];
   }
   int homeGrowth=(tick+d)/2-(tick+1)/2;
   if(incoming>home+homeGrowth)return false;
  }
  return true;
 }
 Candidate defend(){
  Candidate best;int urgent=-1;double urgency=-1;int deficit=0;
  for(int e=0;e<N;e++)if(threatETA[e]<INF){
   int lack=int(threat[e])-army(g)+1;
   // Equal forces already hold the general. A one-troop close margin is
   // useful, but an adequate distant defence is not an urgent campaign.
   if(lack<=0||(lack==1&&threatETA[e]>2))continue;
   double q=(lack+2.0)/(threatETA[e]+1.0);
   if(q>urgency){urgency=q;urgent=e;deficit=lack;}
  }
  bool coalitionDanger=coalitionCount>=2&&coalitionNeed>army(g);
  if(coalitionDanger){
   urgent=coalitionSource;deficit=coalitionNeed-army(g)+1;
   urgency=(deficit+2.0)/(coalitionETA+1.0);
  }
  if(urgent<0)return best;
  bfs(g,fd,true);
  int deadline=coalitionDanger?coalitionDeadline:threatETA[urgent]-(own[urgent]<p?1:0);
  // Reinforcements must arrive before the actual dangerous route does.
  // Credit is capped by the deficit instead of the attacker's raw stack.
  for(int s=0;s<N;s++)if(mine(s)&&s!=g&&canMoveNow(s)&&fd[s]<=deadline){
   int kbest=-1;for(int k=0;k<4;k++){int v=nb[s][k];if(v>=0&&fd[v]+1==fd[s]){if(kbest<0||army(v)>army(nb[s][kbest]))kbest=k;}}
   if(kbest<0)continue;
   int delivered=moveArmy(s);int d=fd[s];
   double score=24+std::min(delivered,deficit)*2.0/(d+.5)+urgency*2;
   int dest=nb[s][kbest];score-=risk(s,dest,delivered);
   if(score>best.score)best={score,s,g,kbest,false};
  }
  // Intercept only an attacker with a credible route to the general.
  for(int s=0;s<N;s++)if(mine(s)&&canMoveNow(s)){for(int k=0;k<4;k++){
   int v=nb[s][k];if(v<0||!o->visible[v]||own[v]<0||own[v]==p)continue;
   bool coalitionMember=coalitionDanger&&closeOwnerSource[own[v]]==v;
   if(!coalitionMember&&(threatETA[v]>7||threat[v]<army(g)))continue;
   bool half=false;if(!canDepart(s,half))continue;
   int n=moveArmy(s,half),enemy=rememberedDefense(v);double score=-1e10;
   if(n>enemy){score=32+std::min(enemy,100)*.5+urgency*3;}
   else if((threatETA[v]<=2||coalitionMember)&&s!=g){
    double remainingDanger=coalitionMember?coalitionNeed-std::min(n,closeOwnerArmy[own[v]]):threat[v]-n;
    if(remainingDanger<army(g))score=45+urgency*3;
   }
   score-=risk(s,v,n);
   if(score>best.score)best={score,s,v,k,half};
  }}
  return best;
 }
 // A short simple path can collect several visible friendly enemy stacks.
 // Enumerating only four moves avoids both speculative long plans and the
 // repeated-army cycles that a negative-cost path formulation would permit.
 Candidate mergerShield(const Candidate &current){
  Candidate best;int cells[64],index[MAXN],links[64][4],count=0,home=-1;
  std::fill(index,index+N,-1);bool contact=false;
  for(int c=0;c<N;c++)if(gd[c]<=4&&ter[c]!=1){
   if(count==64)return best;
   index[c]=count;cells[count]=c;if(c==g)home=count;count++;
   if(o->visible[c]&&own[c]>=0&&own[c]!=p&&army(c)>2)contact=true;
  }
  if(!contact||home<0)return best;
  for(int j=0;j<count;j++)for(int k=0;k<4;k++){
   int v=nb[cells[j]][k];links[j][k]=v>=0?index[v]:-1;
  }
  auto ready=[&](int c){return army(c)+int(own[c]>=0&&(((tick+1)%2==0&&st[c]>0&&st[c]<3)||((tick+1)%50==0&&st[c]==0)));};
  int early[8]{};
  for(int k=0;k<4;k++){int e=nb[g][k];if(e>=0&&o->visible[e]&&own[e]>=0&&own[e]<p)early[own[e]]=std::max(early[own[e]],moveArmy(e));}
  int earlyDamage=0;for(int e=0;e<p;e++)earlyDamage+=early[e];
  auto danger=[&](int source,int dest,int moving,bool merge){
   int guard[64],arrivals[8][5]{};
   bool captured=dest>=0&&!mine(dest)&&moving>rememberedDefense(dest);
   int homeArmy=ready(g)-(source==g?moving:0)+(dest==g?moving:0);
   for(int j=0;j<count;j++){
    int c=cells[j];guard[j]=0;
    if(c!=g){
     if(mine(c))guard[j]=ready(c)-(c==source?moving:0)+(c==dest?moving:0);
     else if(c==dest&&captured)guard[j]=moving-rememberedDefense(c);
     else if(o->visible[c]&&own[c]==-2)guard[j]=army(c);
    }
   }
   auto walk=[&](auto &&self,int u,int force,int steps,std::uint64_t visited,int enemy)->void{
    if(steps==4)return;
    int c=cells[u],d=steps+1;
    if(steps>0&&(((tick+d)%2==0&&st[c]>0&&st[c]<3)||((tick+d)%50==0&&st[c]==0)))force++;
    int sent=force-1;if(sent<=0)return;
    for(int k=0;k<4;k++){
     int v=links[u][k];if(v<0||(visited&(std::uint64_t(1)<<v)))continue;
     int target=cells[v];if(gd[target]>4-d)continue;
     if(v==home){arrivals[enemy][d]=std::max(arrivals[enemy][d],sent);continue;}
     int next=sent-guard[v];
     if(merge&&o->visible[target]&&own[target]==enemy&&!(target==dest&&captured)){
      int extra=0;
      if(st[target]>0&&st[target]<3)extra=(tick+d)/2-(tick+1)/2;
      else if(st[target]==0)extra=(tick+d)/50-(tick+1)/50;
      next+=ready(target)+extra;
     }
     if(next>0)self(self,v,next,d,visited|(std::uint64_t(1)<<v),enemy);
    }
   };
   for(int j=0;j<count;j++){
    int c=cells[j];if(!o->visible[c]||own[c]<0||own[c]==p||ready(c)<2)continue;
    if(c==dest&&captured&&own[c]>p)continue;
    walk(walk,j,ready(c),0,std::uint64_t(1)<<j,own[c]);
   }
   double value=0;int byOwner[8]{};
   if(earlyDamage>ready(g))value=100000.0+10*(earlyDamage-ready(g));
   for(int d=1;d<=4;d++){
    int incoming=0;
    for(int e=0;e<o->players;e++)if(e!=p&&o->alive[e]){byOwner[e]=std::max(byOwner[e],arrivals[e][d]);incoming+=byOwner[e];}
    int lack=incoming-homeArmy-((tick+d)/2-(tick+1)/2);
    if(lack>0)value=std::max(value,100000.0/(d*d)+10.0*lack/d);
   }
   return value;
  };
  double before=danger(-1,-1,0,true);
  if(before<=0||before<=danger(-1,-1,0,false)+1e-8)return best;
  double reference=before;
  if(current.source>=0){int dest=nb[current.source][current.dir];reference=std::min(reference,danger(current.source,dest,moveArmy(current.source,current.half),true));}
  if(reference<=0)return best;
  double bestAfter=reference,bestTie=-1e30;
  for(int j=0;j<count;j++){
   int source=cells[j];if(!mine(source)||!canMoveNow(source))continue;
   bool half=false;if(!canDepart(source,half))continue;int moving=moveArmy(source,half);
   bool lost=false;
   for(int k=0;k<4;k++){int e=nb[source][k];if(e>=0&&o->visible[e]&&own[e]>=0&&own[e]<p&&moveArmy(e)>ready(source))lost=true;}
   if(lost)continue;
   for(int k=0;k<4;k++){
    int dest=nb[source][k];if(dest<0||index[dest]<0||ter[dest]==1)continue;
    if(!mine(dest)&&(st[dest]==2||moving<=rememberedDefense(dest)))continue;
    if(risk(source,dest,moving)>1000)continue;
    double after=danger(source,dest,moving,true);
    if(after+1e-8>=reference)continue;
    double tie=(mine(dest)?0:1)+std::min(moving,1000)*.0001;
    if(after<bestAfter-1e-8||(std::abs(after-bestAfter)<1e-8&&tie>bestTie)){
     bestAfter=after;bestTie=tie;best={500+(before-after)*.002,source,dest,k,half};
    }
   }
  }
  return best;
 }
 int64_t act(const ArenaObservation *obs){
  if(obs->width*obs->height>MAXN||obs->general_pos<0)return 0;
  update(obs);Candidate best;
  int alive=0;for(int e=0;e<o->players;e++)alive+=o->alive[e]!=0;
  // A verified one-step general capture dominates other strategic considerations.
  for(int s=0;s<N;s++)if(mine(s))for(int k=0;k<4;k++){
   int v=nb[s][k];if(v>=0&&st[v]==2&&own[v]>=0&&own[v]!=p&&moveArmy(s)>captureDefense(v)){
    bool half=false;
    if(o->players>=7){
     // Preserve the tested plain-memory capture handling for crowded games.
     if(alive>2&&s==g){
      int needed=1,perOwner[8]{};
      for(int e=0;e<N;e++)if(o->visible[e]&&own[e]>=0&&own[e]!=p&&own[e]!=own[v]&&gd[e]<=2)
       perOwner[own[e]]=std::max(perOwner[own[e]],moveArmy(e)+1-gd[e]);
      int together=0;for(int e=0;e<o->players;e++)together+=perOwner[e];
      needed=std::max(needed,together);
      if(needed>1){half=true;if(moveArmy(s,true)<=captureDefense(v)||moveArmy(s)+1-moveArmy(s,true)<needed)continue;}
     }
    }else if(alive>2&&!captureSafe(s,v,moveArmy(s))){
     half=true;
     if(moveArmy(s,true)<=captureDefense(v)||!captureSafe(s,v,moveArmy(s,true)))continue;
    }
    oldTarget=v;oldSource=v;return arena_move(s,k,half);
   }
  }
  Candidate defense=defend();
  if(o->players<=6){Candidate merged=mergerShield(defense);if(merged.source>=0)defense=merged;}
  // A threatened general takes precedence unless an immediate elimination exists.
  if(defense.score>22)best=defense;
  double income=.5*(std::max<int64_t>(0,o->total_cities[p])+1)+.02*o->total_land[p];
  std::vector<int> sources;for(int i=0;i<N;i++)if(mine(i)&&canMoveNow(i))sources.push_back(i);
  std::sort(sources.begin(),sources.end(),[&](int x,int y){double ax=army(x)+(x==oldSource?5:0),ay=army(y)+(y==oldSource?5:0);return ax>ay;});
  if(sources.size()>16){
   // Preserve the twelve strongest mobile groups. Four further slots keep
   // useful border armies from being crowded out by productive rear cities.
   if(o->players<=6){
    std::vector<int> keep(sources.begin(),sources.begin()+12);
    std::vector<std::pair<int,int>> border;
    border.reserve(sources.size()-12);
    int scout=-1,scoutGain=0,scoutDistance=INF;
    for(int j=12;j<int(sources.size());j++){
     int source=sources[j],key=ed[source],visionGain=0,capturable=0;
     bool half=false;
     if(canDepart(source,half)){
      int moving=moveArmy(source,half);
      for(int k=0;k<4;k++){
       int v=nb[source][k];
       if(v>=0&&ter[v]!=1&&!mine(v)&&moving>rememberedDefense(v)){
        capturable++;
        // Every owned cell already reveals its own 3x3 neighborhood.
        // A capturable neighbor measures actual next-step new visibility.
        visionGain=std::max(visionGain,unknown[v]);
       }
      }
     }
     if(visionGain>0&&(visionGain>scoutGain||
        (visionGain==scoutGain&&ed[source]<scoutDistance))){
      scout=j;scoutGain=visionGain;scoutDistance=ed[source];
     }
     if(key>=INF)key=INF-visionGain*2-capturable;
     border.emplace_back(key,j);
    }
    std::partial_sort(border.begin(),border.begin()+4,border.end());
    // Reserve at most one of the same four slots for direct exploration;
    // preserve all four original border choices when no such move exists.
    int extra[4],extraCount=0;
    if(scout>=0)extra[extraCount++]=scout;
    for(int j=0;j<4&&extraCount<4;j++)
     if(border[j].second!=scout)extra[extraCount++]=border[j].second;
    std::sort(extra,extra+4);
    for(int j:extra)keep.push_back(sources[j]);
    sources.swap(keep);
   }else{
    std::vector<int> keep(sources.begin(),sources.begin()+12);
    std::vector<std::pair<int,int>> border;
    border.reserve(sources.size()-12);
    for(int j=12;j<int(sources.size());j++){
     int source=sources[j],key=ed[source];
     if(key>=INF){
      // Before enemy contact, use new vision and immediately capturable
      // neighboring terrain as the exploration-front substitute.
      key=INF-unknown[source]*2;
      for(int k=0;k<4;k++){
       int v=nb[source][k];
       if(v>=0&&ter[v]!=1&&!mine(v)&&moveArmy(source)>rememberedDefense(v))key--;
      }
     }
     border.emplace_back(key,j);
    }
    std::partial_sort(border.begin(),border.begin()+4,border.end());
    int extra[4];for(int j=0;j<4;j++)extra[j]=border[j].second;
    std::sort(extra,extra+4);
    for(int j:extra)keep.push_back(sources[j]);
    sources.swap(keep);
   }
  }
  bool saveCity=false;
  if(tick>=50&&tick<200&&o->total_land[p]>=22&&ed[g]>8){
   for(int c=0;c<N;c++)if(st[c]==1&&own[c]<0&&gd[c]<=7){saveCity=true;break;}
  }
  for(int s:sources){
   bool half=false;if(!canDepart(s,half))continue;
   int moving=moveArmy(s,half);if(moving<=0)continue;
   // These tactical adjustments depend on the first step, not on the final
   // campaign target. Store hazards individually to preserve subtraction order
   // and therefore the original floating-point tie behavior exactly.
   double stepRisk[4]{},stepHazard[4][4]{};int hazardCount[4]{};
   for(int direction=0;direction<4;direction++){
    int dest=nb[s][direction];if(dest<0)continue;
    stepRisk[direction]=risk(s,dest,moving);
    if(own[dest]!=p){
     int remain=moving-rememberedDefense(dest);
     for(int k=0;k<4;k++){
      int e=nb[dest][k];
      if(e>=0&&e!=s&&o->visible[e]&&own[e]>=0&&own[e]!=p&&a[e]>remain+2)
       stepHazard[direction][hazardCount[direction]++]=std::min(5.,a[e]*.025);
     }
    }
   }
   int depth=24;for(int e=0;e<o->players;e++)if(e!=p&&enemyGeneral[e]>=0&&o->alive[e])depth=32;
   search(s,moving,depth);
   // Iterate reachable targets in the original increasing cell order.
   for(int word=0;word<(N+63)/64;word++){
    std::uint64_t bits=searchTargets[word];
    while(bits){
    int v=word*64+__builtin_ctzll(bits);bits&=bits-1;
    int d=dist[v];if(d>24&&st[v]!=2)continue;
    if(saveCity&&s==g&&st[v]==0)continue;
    int ndef=rememberedDefense(v);bool enemy=own[v]>=0&&own[v]!=p;
    bool city=(st[v]==1), king=(st[v]==2&&enemy);
    bool immediateKing=d==1&&king&&moving>captureDefense(v);
    if(d==1&&king&&!immediateKing)king=false; // Score an uncertain strike as an ordinary attack.
    // The same immediate capture must not re-enter through campaign scoring
    // after the elimination-aware check rejected it above.
    if(o->players<=6&&alive>2&&immediateKing&&!captureSafe(s,v,moving))continue;
    double reward;
    if(king){
      reward=120+std::min<int64_t>(300,o->total_army[own[v]])*.20+std::min<int64_t>(200,o->total_land[own[v]])*.40;
      // Enemy reinforcement during travel is a real obstacle to a naked rush.
      int required=d/2+2;if(d>1&&force[v]<=required)continue;
    }else if(city){
      reward=15.0+std::min(10.0,army(s)*.05)+(enemy?5:0);
      // Reward inexpensive investments with an actual surviving garrison;
      // distant cities do not get a speculative assembly bonus.
      if(d<=4&&force[v]>=8)reward+=3.0+(enemy?2.0:0.0);
      if(force[v]<4)reward*=.72;
    }else{
      reward=1.8+(enemy?2.1:0)+unknown[v]*.22;
      if(ter[v]==255)reward+=.6;
      if(enemy&&ndef>3)reward+=std::min(6.0,ndef*.06);
      // A modest bias toward penetrating a weaker opponent also buys vision
      // of its general, without committing to an unlocated speculative rush.
      if(enemy){
       double advantage=double(o->total_army[p])/std::max<int64_t>(1,o->total_army[own[v]]);
       if(advantage>1.5)reward+=.45+unknown[v]*.07;
       if(advantage>2.5)reward+=.35;
      }
      // Maintain a broad, connected productive domain.
      if(gd[v]<8)reward+=.3;
      if(tick<80)reward+=unknown[v]*.10;
      // Once the opening is established, preserve momentum in a mobile
      // field army instead of spending the action on an exhausted fragment.
      if(o->players<=5&&tick>=140)reward*=.70+.30*std::min(1.0,force[v]/18.0);
      // A strong penetration into fresh enemy terrain can reveal its general;
      // old, fully explored border tiles do not receive this scouting bonus.
      if(o->players<=5&&enemy&&moving>=20&&force[v]>=8)reward+=unknown[v]*.55;
    }
    double lost=pathLoss[v];
    double den=d+.8+lost/(income*2.5+2);
    double score=reward/den;
    if(king)score+=6.0/(1+d*.07);
    if(immediateKing)score+=100000;
    if(v==oldTarget)score*=1.13;
    if(s==oldSource)score*=1.025;
    if(city&&tick<95)score*=.75;
    int direction=first[v];
    // A lethal departure is infeasible; a numerical king bonus must not
    // cancel its risk penalty. Preserve immediate last-opponent captures.
    if(o->players<=6&&stepRisk[direction]>=1000&&!(alive<=2&&immediateKing))continue;
    score-=stepRisk[direction];
    if(!king)for(int k=0;k<hazardCount[direction];k++)score-=stepHazard[direction][k];
    if(score>best.score)best={score,s,v,first[v],half};
    }
   }
  }
  // Mobilize productive interiors: marginal delivery per action, directed toward
  // a persistent campaign or the nearest unowned frontier.
  bool protectInvestment=false;
  if(best.source>=0&&best.score<22){
   int next=nb[best.source][best.dir];
   int moving=moveArmy(best.source,best.half);
   if(next>=0&&!mine(next)&&st[next]==1&&moving>=rememberedDefense(next)+8)
    protectInvestment=true;
   if(best.target==oldTarget&&best.source==oldSource&&st[best.target]==1){
    int distance=std::abs(best.source/W-best.target/W)+std::abs(best.source%W-best.target%W);
    if(distance<=6&&moving>=rememberedDefense(best.target)+12)protectInvestment=true;
   }
  }
  if(best.source>=0&&best.score<22&&!protectInvestment){
   int goal=best.target;
   bfs(goal,targetDist,false,true);
   for(int s:sources){if(saveCity&&s==g)continue;bool half=false;if(!canDepart(s,half))continue;int n=moveArmy(s,half);if(n<3)continue;
    for(int k=0;k<4;k++){int v=nb[s][k];if(v<0||!mine(v)||targetDist[v]>=targetDist[s]||targetDist[s]>=INF)continue;
      // Gathering is useful when it shortens the time to an attainable city or assault.
      double score=(std::sqrt(double(n))*.38)/(1+targetDist[v]*.11);
      if(st[s]>0)score*=1.20;
      if(ed[v]<ed[s])score*=1.15;
      if(v==oldSource)score*=1.20;
      if(targetDist[s]>15)score*=.65;
      score-=risk(s,v,n);
      if(score>best.score)best={score,s,goal,k,half};
    }
   }
  }
  if(best.source<0||best.score<0.10)return 0;
  oldTarget=best.target;oldSource=nb[best.source][best.dir];lastTarget=oldSource;
  return arena_move(best.source,best.dir,best.half);
 }
};
}
extern "C" void *arena_create(int player,std::uint64_t seed,const char*) { return new Policy(player,seed); }
extern "C" std::int64_t arena_act(void *instance,const ArenaObservation *observation) { return static_cast<Policy*>(instance)->act(observation); }
extern "C" void arena_destroy(void *instance) { delete static_cast<Policy*>(instance); }
