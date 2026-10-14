// Razor: bounded route planning with explicit troop-assembly plans.
// All persistent knowledge is derived from this seat's observations. The
// tactical safety and observation layer is adapted from our kestrel series.
#include "arena_policy.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

namespace {
constexpr int MAXN=4096, INF=1000000;
double parameter(const char *json,const char *name,double fallback){
 if(!json)return fallback;
 const char *key=std::strstr(json,name);if(!key)return fallback;
 const char *colon=std::strchr(key,':');if(!colon)return fallback;
 char *end=nullptr;double value=std::strtod(colon+1,&end);
 return end!=colon+1&&std::isfinite(value)?value:fallback;
}
struct Candidate { double score=-1e30; int source=-1, target=-1, dir=0; bool half=false; };
struct Route {
 double gain=0,rank=0,loss=0,hazard=0,exposure=0;
 int cell=-1,source=-1,force=0,dir=-1,steps=0;
 bool half=false;
 short path[13]{};
};
struct Policy {
 int p,W=0,H=0,N=0,g=-1,tick=0,oldTarget=-1,oldSource=-1,lastTarget=-1,aliveCount=0;
 uint64_t rng;
 double harvestValue=0;
 bool harvestContact=false,harvestIncome=false;
 bool growthBeforeMove=true;
 int growthEvidence=0,receiptSource=-1,receiptTarget=-1,receiptTick=-1;
 int receiptArmy=0,receiptTargetArmy=0,receiptOwner=-1;
 bool receiptHalf=false;

 int ter[MAXN],st[MAXN],own[MAXN],seen[MAXN],a[MAXN],lastOwn[MAXN];
 int nb[MAXN][4],gd[MAXN],fd[MAXN],ed[MAXN],unknown[MAXN],fogGain[MAXN],reserve[MAXN],targetDist[MAXN];
 int dist[MAXN],force[MAXN],pathLoss[MAXN],first[MAXN],parent[MAXN],queue[MAXN];
 double threat[MAXN],pathExposure[MAXN],weightedExposure[MAXN];
 int weightedParent[MAXN];
 int threatETA[MAXN], attackCost[MAXN], nextAttackCost[MAXN],attackGuard[MAXN];
 int homeCells[MAXN],homeCount=0;
 std::uint64_t searchTargets[MAXN/64];
 int rememberedReserve=1, rememberedReserveTick=-1000;
 int closeAttackers[20], closeAttackerCount=0;
 int closeOwnerArmy[8], closeOwnerETA[8], closeOwnerSource[8];
 int coalitionNeed=0, coalitionETA=INF, coalitionDeadline=INF, coalitionSource=-1, coalitionCount=0;
 int enemyGeneral[8], oldTotals[8]{}; double growth[8]{};
 int armyCache[MAXN],fullMoveCache[MAXN],halfMoveCache[MAXN],defenseCache[MAXN],routeDelta[MAXN];
 std::vector<int> sources,keep;
 std::vector<std::pair<int,int>> border;
 double weightedCost[MAXN],heapCost[MAXN*4];
 int weightedForce[MAXN],weightedSteps[MAXN],weightedLoss[MAXN],weightedFirst[MAXN],heapCell[MAXN*4];
 bool weightedDone[MAXN];
 bool geometryReady=false;
 Route beam[96],expanded[384];
 double cellRank[MAXN];
 int treeParent[MAXN],treeOrder[MAXN],treeCount=0;
 int gatherForce[MAXN][13],gatherSource[MAXN][13],gatherDir[MAXN][13];
 const ArenaObservation *o=nullptr;
 Policy(int player,uint64_t seed,const char *params):p(player),rng(seed) {
  harvestValue=std::clamp(parameter(params,"\"harvest_value\"",0),0.0,4.0);
  harvestContact=parameter(params,"\"harvest_contact\"",0)>0;
  harvestIncome=parameter(params,"\"harvest_income\"",0)>0;
  std::fill(ter,ter+MAXN,255);std::fill(st,st+MAXN,255);std::fill(own,own+MAXN,-3);
  std::fill(a,a+MAXN,0);std::fill(seen,seen+MAXN,-10000);std::fill(lastOwn,lastOwn+MAXN,-3);
  std::fill(enemyGeneral,enemyGeneral+8,-1);
  sources.reserve(MAXN);keep.reserve(MAXN);border.reserve(MAXN);
 }
 uint64_t random(){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;return rng;}
 // terrain=2 is an observation-only fog obstacle (mountain or city).
 bool blocked(int i)const{return ter[i]==1||ter[i]==2;}
 bool mine(int i)const{return i>=0 && o->owner[i]==p;}
 int army(int i)const{return armyCache[i];}
 int moveArmy(int i,bool half=false)const {return half?halfMoveCache[i]:fullMoveCache[i];}
 bool canMoveNow(int i)const {return moveArmy(i)>0;}
 int actionTurn(int depth)const{return tick+depth-int(!growthBeforeMove);}
 int futureGrowth(int depth,int period=2)const{return std::max(0,actionTurn(depth)/period-tick/period);}
 // Infer growth/action order from isolated, fully observed transfers. The
 // local engine grows before moving; an online server can grow afterwards.
 // No backend identity or hidden state is needed, and ambiguous receipts do
 // not change the model (network delays can skip or postpone an action).
 void learnGrowth(const ArenaObservation *obs){
  if(receiptSource<0||obs->tick!=receiptTick+1)return;
  int s=receiptSource,t=receiptTarget;
  if(!obs->visible[s]||!obs->visible[t]||obs->owner[s]!=p)return;
  bool matches[2]{};
  for(int before=0;before<2;before++){
   int ready=receiptArmy+before;
   int sent=receiptHalf?ready/2:std::max(0,ready-1);
   int left=ready-sent+!before;
   int ownerAfter=(sent||receiptOwner==p)?p:receiptOwner;
   matches[before]=obs->army[s]==left&&obs->army[t]==receiptTargetArmy+sent&&obs->owner[t]==ownerAfter;
  }
  if(matches[0]==matches[1])return;
  growthEvidence=std::clamp(growthEvidence+(matches[1]?-1:1),-4,4);
  if(growthEvidence>=2)growthBeforeMove=false;
  if(growthEvidence<=-2)growthBeforeMove=true;
 }
 void rememberAction(const ArenaObservation *obs,std::int64_t encoded){
  receiptSource=-1;
  if(encoded<=0)return;
  int s=int((encoded-1)/8),dir=int((encoded-1)%8)/2;
  if(s<0||s>=N||obs->owner[s]!=p)return;
  int t=nb[s][dir];
  if(t<0||!obs->visible[t]||obs->structure[s]<1||obs->structure[s]>2||
     obs->structure[t]!=0||(tick+1)%2||(tick+1)%50==0)return;
  if(obs->owner[t]!=p&&(obs->owner[t]!=-1||obs->army[t]!=0))return;
  for(int c:{s,t})for(int k=0;k<4;k++){
   int e=nb[c][k];if(e>=0&&obs->visible[e]&&obs->owner[e]>=0&&obs->owner[e]!=p)return;
  }
  if(obs->army[s]>10000000||obs->army[t]>10000000)return;
  receiptSource=s;receiptTarget=t;receiptTick=tick;
  receiptArmy=int(obs->army[s]);receiptTargetArmy=int(obs->army[t]);
  receiptOwner=obs->owner[t];receiptHalf=(encoded-1)%2;
 }
 void bfs(int root,int *out,bool friends=false,bool noCities=false,bool fogPass=false){
  std::fill(out,out+N,INF);if(root<0)return;int h=0,z=0;queue[z++]=root;out[root]=0;
  while(h<z){int u=queue[h++];for(int k=0;k<4;k++){int v=nb[u][k];if(v<0||ter[v]==1||(!fogPass&&ter[v]==2)||out[v]!=INF||(friends&&!mine(v))||(noCities&&st[v]==1&&!mine(v)))continue;out[v]=out[u]+1;queue[z++]=v;}}
 }
 // Cache exactly the cells visited by the original twelve-layer defense DP.
 void cacheHomeGeometry(){
  homeCount=0;
  for(int c=0;c<N;c++)if(gd[c]<=12)homeCells[homeCount++]=c;
 }
 void update(const ArenaObservation *obs){
  learnGrowth(obs);
  o=obs;tick=int(o->tick);g=o->general_pos;
  aliveCount=0;for(int e=0;e<o->players;e++)aliveCount+=o->alive[e]!=0;
  if(W==0){W=o->width;H=o->height;N=W*H;for(int i=0;i<N;i++){int x=i%W,y=i/W;nb[i][0]=y?i-W:-1;nb[i][1]=x+1<W?i+1:-1;nb[i][2]=y+1<H?i+W:-1;nb[i][3]=x?i-1:-1;}}
  bool mapChanged=!geometryReady;
  for(int i=0;i<N;i++)armyCache[i]=int(std::min<int64_t>(10000000,std::max<int64_t>(0,o->army[i])));
  for(int i=0;i<N;i++){
   if(o->visible[i]&&ter[i]!=o->terrain[i])mapChanged=true;
   if(!o->visible[i]&&o->terrain[i]==2&&ter[i]==255){ter[i]=2;mapChanged=true;}
   lastOwn[i]=own[i];if(o->visible[i]){ter[i]=o->terrain[i];st[i]=o->structure[i];own[i]=o->owner[i];a[i]=army(i);seen[i]=tick;if(st[i]==2&&own[i]>=0&&own[i]!=p)enemyGeneral[own[i]]=i;}
   else if(own[i]==p){own[i]=-3;a[i]=0;}
   else if(own[i]>=0&&!o->alive[own[i]]){own[i]=-3;a[i]=std::max(1,a[i]/2);if(st[i]==2)st[i]=1;}
   reserve[i]=1;threat[i]=0;threatETA[i]=INF;
  }
  // Observation and remembered terrain are fixed for this decision. Keep the
  // original growth-before-action expressions, evaluated once per cell.
  for(int i=0;i<N;i++){
   int n=armyCache[i];
   if(growthBeforeMove&&(tick+1)%2==0&&st[i]>0&&st[i]<3)n++;
   else if(growthBeforeMove&&(tick+1)%50==0&&st[i]==0)n++;
   fullMoveCache[i]=std::max(0,n-1);halfMoveCache[i]=n/2;
   if(mine(i))defenseCache[i]=0;
   else if(own[i]<0)defenseCache[i]=std::max(0,a[i]);
   else{
    int add=(st[i]>0&&st[i]<3)?std::max(0,tick-seen[i])/2:0;
    defenseCache[i]=a[i]+std::min(add,200)+int(growthBeforeMove&&o->visible[i]&&(((tick+1)%2==0&&st[i]>0&&st[i]<3)||((tick+1)%50==0&&st[i]==0)));
    if(!o->visible[i]){
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
  if(mapChanged){bfs(g,gd,false,false,true);cacheHomeGeometry();}
  std::fill(ed,ed+N,INF);int h=0,z=0;
  for(int i=0;i<N;i++)if(o->visible[i]&&own[i]>=0&&own[i]!=p){ed[i]=0;queue[z++]=i;}
  while(h<z){int u=queue[h++];for(int k=0;k<4;k++){int v=nb[u][k];if(v>=0&&ter[v]!=1&&ed[v]==INF){ed[v]=ed[u]+1;queue[z++]=v;}}}
  for(int i=0;i<N;i++){
   fogGain[i]=0;int x=i%W,y=i/W;
   if(mapChanged)unknown[i]=0;
   for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){
    int xx=x+dx,yy=y+dy;
    if(xx>=0&&xx<W&&yy>=0&&yy<H){int c=yy*W+xx;if(!o->visible[c]&&!blocked(c))fogGain[i]++;if(mapChanged&&ter[c]==255)unknown[i]++;}
   }
  }
  geometryReady=true;
  // Minimum troop loss of an enemy route to our general, up to 12 moves.
  // Our current blocking armies and neutral city garrisons really do absorb
  // attackers. Third-party troops are not trusted as permanent protection.
  // Costs are positive, so considering walks cannot create free army cycles.
  std::fill(closeOwnerArmy,closeOwnerArmy+8,0);
  std::fill(closeOwnerETA,closeOwnerETA+8,INF);
  std::fill(closeOwnerSource,closeOwnerSource+8,-1);
  // The DP can only affect visible mobile enemies in the cached home area.
  // Retain the normal zeroed threat state when no such enemy exists.
  bool hasHomeThreat=false;
  if(ed[g]<=12)for(int j=0;j<homeCount;j++){
   int c=homeCells[j];
   if(o->visible[c]&&own[c]>=0&&own[c]!=p&&a[c]>1){hasHomeThreat=true;break;}
  }
  if(hasHomeThreat){
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
     int kingGrowth=futureGrowth(d);
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
   coalitionNeed=std::max(0,combinedArrival-futureGrowth(lastArrival));
   coalitionDeadline=coalitionETA;
   int beforeOurTurn=0;
   for(int e=0;e<p;e++)if(closeOwnerETA[e]==coalitionETA)beforeOurTurn+=closeOwnerArmy[e];
   // A harmless early-seat hit still leaves time for our same-tick relief;
   // shorten the deadline only when that earlier prefix is itself lethal.
   if(beforeOurTurn>army(g)+futureGrowth(coalitionETA))coalitionDeadline--;
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
  if(ed[g]>10&&need<=1&&retained<=1)reserve[g]=1;

  for(int i=0;i<N;i++){
   if(mine(i)){
    routeDelta[i]=std::max(0,army(i)-reserve[i]+1);
    if(i==g&&reserve[i]>1)routeDelta[i]=0;
   }else routeDelta[i]=-defenseCache[i]-(ter[i]==255?1:0);
  }
 }
 double eliminationValue(int e)const {
  if(o->players==2)return 120+std::min<int64_t>(300,o->total_army[e])*.20+
                                   std::min<int64_t>(200,o->total_land[e])*.40;
  double soldiers=std::max<int64_t>(0,o->total_army[e]);
  double land=std::max<int64_t>(0,o->total_land[e]);
  return 120+.20*std::sqrt(soldiers*std::min(300.,soldiers))+
             .40*std::sqrt(land*std::min(200.,land));
 }
 int rememberedDefense(int i)const {return defenseCache[i];}
 bool canDepart(int s,bool &half)const{
  half=false;int n=moveArmy(s)+1;if(n<2)return false;
  if(reserve[s]>1){if(n-n/2>=reserve[s])half=true;else return false;}
  return true;
 }
 // Full departure can expose a productive city even when half departure
 // leaves an ample garrison. Evaluate both legal choices in that case.
 bool usefulCityHalf(int s)const {
  if(st[s]!=1||moveArmy(s,true)<2)return false;
  int remaining=moveArmy(s)+1-moveArmy(s,true);
  for(int k=0;k<4;k++){
   int e=nb[s][k];
   if(e>=0&&o->visible[e]&&own[e]>=0&&own[e]!=p&&
      moveArmy(e)>2&&moveArmy(e)<=remaining)return true;
  }
  return false;
 }
 // Positioning an assembled army must not be rewarded more merely because
 // the next square is friendly. Use one value on both sides of the frontier,
 // so the final attack completes the purpose of the preceding transport.
 double deploymentValue(int s,int dest,int moving,int distance)const {
  double value=std::sqrt(double(moving))*.38/(1+distance*.11);
  if(st[s]>0)value*=1.20;
  if(ed[dest]<ed[s])value*=1.15;
  if(dest==oldSource)value*=1.20;
  if(distance+1>15)value*=.65;
  return value;
 }
 // Price a counterattack along the whole proposed path. Charging it only
 // on step one lets a receding-horizon planner postpone the same hazard
 // forever by walking away and then back. A captured path cell cannot also
 // supply a hostile counterattack in that plan.
 double landingExposure(int from,int dest,int remaining,const int *parents)const {
  if(mine(dest))return 0;
  double exposure=0;
  for(int k=0;k<4;k++){
   int e=nb[dest][k];
   if(e<0||e==from||!o->visible[e]||own[e]<0||own[e]==p||a[e]<=remaining+2)continue;
   bool captured=false;
   for(int c=from;c>=0;c=parents[c])if(c==e){captured=true;break;}
   if(!captured)exposure+=std::min(5.,a[e]*.025);
  }
  return exposure;
 }
 // Fast bounded breadth-first campaigns, keeping the strongest shortest route.
 void search(int s,int initial,int depth){
  std::fill(dist,dist+N,INF);std::fill(force,force+N,-INF);
  std::fill(searchTargets,searchTargets+(N+63)/64,0);int h=0,z=0;
  queue[z++]=s;dist[s]=0;force[s]=initial;pathLoss[s]=0;first[s]=-1;parent[s]=-1;pathExposure[s]=0;
  while(h<z){int u=queue[h++];if(dist[u]>=depth)continue;
   for(int k=0;k<4;k++){int v=nb[u][k];if(v<0||blocked(v)||v==s)continue;int d=dist[u]+1;if(dist[v]<d)continue;
    int f=force[u];if(u!=s)f-=1;
    f+=routeDelta[v];
    if(f<1)continue;
    if(dist[v]==INF){
     dist[v]=d;queue[z++]=v;
     if(!mine(v))searchTargets[v>>6]|=std::uint64_t(1)<<(v&63);
    }
    int spent=pathLoss[u]+int(u!=s)+(mine(v)?0:rememberedDefense(v)+int(ter[v]==255));
    if(f>force[v]||(f==force[v]&&spent<pathLoss[v])){force[v]=f;pathLoss[v]=spent;first[v]=(u==s?k:first[u]);parent[v]=u;pathExposure[v]=std::max(pathExposure[u],landingExposure(u,v,f,parent));}
   }
  }
 }
 // Positive route costs permit detours around expensive cities without cycles.
 // Duels use this complement only for the largest mobile group.
 Candidate weightedCampaign(int s,double income,bool split=false){
  Candidate best;bool half=false;if(!canDepart(s,half))return best;
  half=half||split;
  int initial=moveArmy(s,half);if(initial<12)return best;
  double scale=income*2.5+2;
  std::fill(weightedCost,weightedCost+N,1e30);std::fill(weightedDone,weightedDone+N,false);
  int heapSize=0;
  auto push=[&](int cell,double cost){
   int j=heapSize++;
   while(j>0){int par=(j-1)/2;if(heapCost[par]<cost||(heapCost[par]==cost&&heapCell[par]<=cell))break;
    heapCost[j]=heapCost[par];heapCell[j]=heapCell[par];j=par;}
   heapCost[j]=cost;heapCell[j]=cell;
  };
  weightedCost[s]=0;weightedForce[s]=initial;weightedSteps[s]=0;weightedLoss[s]=0;weightedFirst[s]=-1;weightedParent[s]=-1;weightedExposure[s]=0;push(s,0);
  while(heapSize){
   int u=heapCell[0];double cost=heapCost[0];--heapSize;
   if(heapSize){int cell=heapCell[heapSize];double last=heapCost[heapSize];int j=0;
    while(j*2+1<heapSize){int child=j*2+1;
     if(child+1<heapSize&&(heapCost[child+1]<heapCost[child]||(heapCost[child+1]==heapCost[child]&&heapCell[child+1]<heapCell[child])))child++;
     if(last<heapCost[child]||(last==heapCost[child]&&cell<=heapCell[child]))break;
     heapCost[j]=heapCost[child];heapCell[j]=heapCell[child];j=child;}
    heapCost[j]=last;heapCell[j]=cell;
   }
   if(weightedDone[u]||cost!=weightedCost[u])continue;
   weightedDone[u]=true;if(weightedSteps[u]>=32)continue;
   for(int k=0;k<4;k++){
    int v=nb[u][k];if(v<0||v==s||blocked(v)||weightedDone[v])continue;
    int d=weightedSteps[u]+1,detach=int(u!=s),nextForce=weightedForce[u]-detach+routeDelta[v];if(nextForce<1)continue;
    int loss=weightedLoss[u]+detach+(mine(v)?0:rememberedDefense(v)+int(ter[v]==255));double next=d+loss/scale;
    if(next>weightedCost[v]||(next==weightedCost[v]&&nextForce<=weightedForce[v]))continue;
    weightedCost[v]=next;weightedForce[v]=nextForce;weightedSteps[v]=d;weightedLoss[v]=loss;
    weightedFirst[v]=u==s?k:weightedFirst[u];weightedParent[v]=u;weightedExposure[v]=std::max(weightedExposure[u],landingExposure(u,v,nextForce,weightedParent));push(v,next);
   }
  }
  for(int v=0;v<N;v++)if(!mine(v)&&weightedCost[v]<1e29){
   bool enemy=own[v]>=0&&own[v]!=p,city=st[v]==1,king=st[v]==2&&enemy;if(!city&&!king)continue;
   int d=weightedSteps[v],direction=weightedFirst[v],remaining=weightedForce[v];if(direction<0||(city&&d>24))continue;
   bool immediateKing=king&&d==1&&initial>captureDefense(v);if(king&&d==1&&!immediateKing)continue;
   if(king&&d>1&&remaining<=d/2+2)continue;
   double reward;
   if(king)reward=eliminationValue(own[v]);
   else{reward=15+std::min(10.,army(s)*.05)+(enemy?5:0);if(d<=4&&remaining>=8)reward+=3+(enemy?2:0);if(remaining<4)reward*=.72;}
   double score=reward/(d+.8+weightedLoss[v]/scale);if(king)score+=6./(1+d*.07);if(immediateKing)score+=100000;
   if(v==oldTarget)score*=(o->players==2&&ed[s]<=8?1.05:1.13);if(s==oldSource)score*=1.025;if(city&&tick<95)score*=.75;
   int dest=nb[s][direction];double departure=risk(s,dest,initial);if(departure>=1000&&!immediateKing)continue;
   if(!mine(dest)){
    int hold=initial-rememberedDefense(dest);bool counter=false;
    for(int k=0;k<4;k++){
     int e=nb[dest][k];if(e<0||e==s||!o->visible[e]||own[e]<0||own[e]==p)continue;
     if(st[dest]==1&&hold>0){int attack=moveArmy(e),guard=hold;
      if(own[e]<p){attack+=int((actionTurn(2)%2==0&&st[e]>0&&st[e]<3)||(actionTurn(2)%50==0&&st[e]==0));guard+=int(actionTurn(2)%2==0);}
      if(attack>guard+2)counter=true;}
    }
    if(counter)score*=.35;score-=departure;
    
   }else score-=departure;
   if(!king)score-=weightedExposure[v];
   if(score>best.score)best={score,s,v,direction,half};
  }
  return best;
 }
 double risk(int source,int dest,int moving)const {
  double cost=0;int remain=moveArmy(source)+1-moving;
  // A screen next to the general cannot be withdrawn while a hostile stack
  // can cross it and kill home before another friendly action can arrive.
  if(source!=g&&gd[source]==1&&dest!=g){
   int home=army(g)+futureGrowth(2);
   for(int k=0;k<4;k++){
    int e=nb[source][k];if(e<0||!o->visible[e]||own[e]<0||own[e]==p)continue;
    if(e==dest&&moving>rememberedDefense(e))continue;
    if(moveArmy(e)-remain-1>home)return 100000;
   }
  }
  if(source==g){
   int garrison=tick>100?std::min(70,std::max(3+tick/60,4+int(o->total_army[p]/25))):1;
   if(ed[g]>10&&reserve[g]==1)garrison=1;
   if(remain<garrison)return 100000;
   for(int j=0;j<closeAttackerCount;j++){
    int e=closeAttackers[j];if(threat[e]<=remain)continue;
    if(e==dest&&moving>rememberedDefense(e))continue;
    return 100000;
   }
  }
  for(int k=0;k<4;k++){int e=nb[source][k];if(e>=0&&e!=dest&&o->visible[e]&&own[e]>=0&&own[e]!=p&&a[e]>remain+1){
    if(source==g)return 100000;
    if(st[source]==1)cost=std::max(cost,5.);
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
  auto ready=[&](int c){return army(c)+(growthBeforeMove&&own[c]>=0?grows(c,tick+1):0);};
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
    int v=nb[e][k];if(v<0||blocked(v))continue;
    if(v==g){
     arriving[enemy][1]=std::max(arriving[enemy][1],sent);
     arriving[enemy][2]=std::max(arriving[enemy][2],sent+grows(e,actionTurn(2)));
    }else if(gd[v]==1){
     int next=sent-guard(v);
     if(o->visible[v]&&own[v]==enemy)next+=ready(v);
     if(next<=0)continue;
     int attack=next+grows(v,actionTurn(2))-1;
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
   int homeGrowth=actionTurn(d)/2-actionTurn(1)/2;
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
 // A pair of visible enemy groups can merge into a raid before either group
 // is individually lethal. Recall a mobile defender while it can still reach
 // home; these are observations of troops, not assumptions about an opponent.
 Candidate forecastDefend(){
  Candidate best;
  if(tick<80||ed[g]>6)return best;
  int top[8][3]{},cell[8][3];
  for(int e=0;e<8;e++)cell[e][0]=cell[e][1]=cell[e][2]=-1;
  for(int c=0;c<N;c++)if(o->visible[c]&&own[c]>=0&&own[c]!=p&&o->alive[own[c]]&&gd[c]<=6){
   int e=own[c],n=moveArmy(c);
   if(n>top[e][0]){
    top[e][2]=top[e][1];cell[e][2]=cell[e][1];
    top[e][1]=top[e][0];cell[e][1]=cell[e][0];top[e][0]=n;cell[e][0]=c;
   }else if(n>top[e][1]){
    top[e][2]=top[e][1];cell[e][2]=cell[e][1];top[e][1]=n;cell[e][1]=c;
   }else if(n>top[e][2]){top[e][2]=n;cell[e][2]=c;}
  }
  int deficit=0,deadline=INF;
  for(int e=0;e<o->players;e++)if(e!=p&&cell[e][0]>=0){
   int first=cell[e][0],second=cell[e][1],third=cell[e][2];
   bool merge=second>=0&&top[e][1]>=7&&
    std::abs(first/W-second/W)+std::abs(first%W-second%W)<=5;
   bool mergeThird=merge&&third>=0&&top[e][2]>=10&&
    std::min(std::abs(first/W-third/W)+std::abs(first%W-third%W),
             std::abs(second/W-third/W)+std::abs(second%W-third%W))<=5;
   int potential=top[e][0]+(merge?top[e][1]:0)+(mergeThird?top[e][2]:0);
   int nearest=std::min(gd[first],std::min(merge?gd[second]:INF,mergeThird?gd[third]:INF));
   int eta=std::max(gd[first],merge?gd[second]:0)+(mergeThird?1:0);
   int needed=potential-army(g)-eta/2;
   if(needed>12&&nearest<=3&&eta<=8&&needed>deficit){
    deficit=needed;deadline=eta;
   }
  }
  if(deficit<=0||deadline<=1)return best;
  bfs(g,fd,true);
  for(int s=0;s<N;s++)if(mine(s)&&s!=g&&fd[s]<=deadline&&fd[s]<=5&&canMoveNow(s)){
   bool half=false;if(!canDepart(s,half))continue;
   int moving=moveArmy(s,half);if(moving<10)continue;
   int kbest=-1;
   for(int k=0;k<4;k++){
    int v=nb[s][k];
    if(v>=0&&fd[v]+1==fd[s]&&(kbest<0||army(v)>army(nb[s][kbest])))kbest=k;
   }
   if(kbest<0)continue;
   int v=nb[s][kbest];
   double score=23+1.2*std::min(moving,deficit+15)/(fd[s]+.5)+8.0/(deadline+1);
   score-=risk(s,v,moving);
   if(score>best.score)best={score,s,g,kbest,half};
  }
  return best;
 }
 // A short simple path can collect several visible friendly enemy stacks.
 // Enumerating only four moves avoids both speculative long plans and the
 // repeated-army cycles that a negative-cost path formulation would permit.
 Candidate mergerShield(const Candidate &current){
  Candidate best;int cells[64],index[MAXN],links[64][4],count=0,home=-1;
  std::fill(index,index+N,-1);bool contact=false;
  for(int c=0;c<N;c++)if(gd[c]<=4&&!blocked(c)){
   if(count==64)return best;
   index[c]=count;cells[count]=c;if(c==g)home=count;count++;
   if(o->visible[c]&&own[c]>=0&&own[c]!=p&&army(c)>2)contact=true;
  }
  if(!contact||home<0)return best;
  for(int j=0;j<count;j++)for(int k=0;k<4;k++){
   int v=nb[cells[j]][k];links[j][k]=v>=0?index[v]:-1;
  }
  auto ready=[&](int c){return army(c)+int(growthBeforeMove&&own[c]>=0&&(((tick+1)%2==0&&st[c]>0&&st[c]<3)||((tick+1)%50==0&&st[c]==0)));};
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
    if(steps>0&&((actionTurn(d)%2==0&&st[c]>0&&st[c]<3)||(actionTurn(d)%50==0&&st[c]==0)))force++;
    int sent=force-1;if(sent<=0)return;
    for(int k=0;k<4;k++){
     int v=links[u][k];if(v<0||(visited&(std::uint64_t(1)<<v)))continue;
     int target=cells[v];if(gd[target]>4-d)continue;
     if(v==home){arrivals[enemy][d]=std::max(arrivals[enemy][d],sent);continue;}
     int next=sent-guard[v];
     if(merge&&o->visible[target]&&own[target]==enemy&&!(target==dest&&captured)){
      int extra=0;
      if(st[target]>0&&st[target]<3)extra=actionTurn(d)/2-actionTurn(1)/2;
      else if(st[target]==0)extra=actionTurn(d)/50-actionTurn(1)/50;
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
    int lack=incoming-homeArmy-(actionTurn(d)/2-actionTurn(1)/2);
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
    int dest=nb[source][k];if(dest<0||index[dest]<0||blocked(dest))continue;
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
 // A single local transfer can preserve an already productive city. This is
 // deliberately below general defence and strong attacks in the action score.
 Candidate cityRelief() {
  Candidate best;
  if(aliveCount!=2)return best;
  for(int city=0;city<N;city++)if(mine(city)&&st[city]==1){
   int hold=moveArmy(city)+1,attack=0;
   for(int k=0;k<4;k++){
    int e=nb[city][k];
    if(e>=0&&o->visible[e]&&own[e]>=0&&own[e]!=p)
     attack=std::max(attack,moveArmy(e));
   }
   if(attack<=hold)continue;
   for(int k=0;k<4;k++){
    int source=nb[city][k];if(source<0||!mine(source))continue;
    bool half=false;if(!canDepart(source,half))continue;
    // An earlier opponent could reduce this source before its selected move.
    // Use only an unexposed donor; the city itself may be recaptured safely.
    bool exposed=false;
    for(int j=0;j<4;j++){
     int e=nb[source][j];
     if(e>=0&&o->visible[e]&&own[e]>=0&&own[e]<p&&moveArmy(e)>0){exposed=true;break;}
    }
    if(exposed)continue;
    if(!half&&hold+moveArmy(source,true)>=attack+2)half=true;
    int moving=moveArmy(source,half);
    if(moving<=0||hold+moving<attack+2)continue;
    double departure=risk(source,city,moving);
    if(departure>=1000)continue;
    double score=4.25-std::min(.20,moving*.001)-departure;
    int dir=(k+2)%4;
    if(score>best.score)best={score,source,city,dir,half};
   }
  }
  return best;
 }
 // Mature domains contain hundreds of one- to ten-troop tiles. If the three
 // largest field stacks hold little of our army, redirect one useful interior
 // stack toward a real enemy front instead of adding another thin frontier.
 Candidate fieldRally(const Candidate &current) {
  Candidate best;
  if(tick<125||o->total_land[p]<35)return best;
  if(current.source>=0&&current.target>=0&&
     st[current.target]==1&&!mine(current.target)&&current.score>=2.5){
   int distance=std::abs(current.source/W-current.target/W)+
                std::abs(current.source%W-current.target%W);
   if(distance<=5&&moveArmy(current.source)>=rememberedDefense(current.target)+4)
    return best;
  }

  int field=0,largest[3]{},anchor=-1;
  for(int s=0;s<N;s++)if(mine(s)&&s!=g){
   int n=army(s);field+=n;
   if(anchor<0||n>army(anchor))anchor=s;
   for(int j=0;j<3;j++)if(n>largest[j]){
    for(int k=2;k>j;k--)largest[k]=largest[k-1];
    largest[j]=n;break;
   }
  }
  if(anchor<0||field<100||4*(largest[0]+largest[1]+largest[2])>=field)return best;
  int goal=-1,knownGeneral=-1;
  for(int e=0;e<o->players;e++)if(e!=p&&o->alive[e]&&enemyGeneral[e]>=0){
   int c=enemyGeneral[e];
   if(knownGeneral<0||std::abs(c/W-anchor/W)+std::abs(c%W-anchor%W)<
                       std::abs(knownGeneral/W-anchor/W)+std::abs(knownGeneral%W-anchor%W))
    knownGeneral=c;
  }
  if(knownGeneral>=0)goal=knownGeneral;
  else if(current.target>=0&&o->visible[current.target]&&
          own[current.target]>=0&&own[current.target]!=p&&st[current.target]==0)
   goal=current.target;
  else{
   int nearest=INF;
   for(int c=0;c<N;c++)if(o->visible[c]&&own[c]>=0&&own[c]!=p&&st[c]==0){
    int distance=std::abs(c/W-anchor/W)+std::abs(c%W-anchor%W);
    if(distance<nearest){nearest=distance;goal=c;}
   }
  }
  if(goal<0)return best;
  bfs(goal,targetDist,false,true);
  for(int s=0;s<N;s++)if(mine(s)&&targetDist[s]>0&&targetDist[s]<=20){
   bool half=false;if(!canDepart(s,half))continue;
   int sent=moveArmy(s,half);if(sent<5)continue;
   for(int dir=0;dir<4;dir++){
    int dest=nb[s][dir];
    if(dest<0||!mine(dest)||targetDist[dest]+1!=targetDist[s])continue;
    if(ed[s]<=1&&army(dest)<army(s))continue;
    double departure=risk(s,dest,sent);
    if(departure>=1000)continue;
    int landing=moveArmy(dest)+1+sent;
    bool exposed=false;
    for(int k=0;k<4;k++){
     int e=nb[dest][k];
     if(e>=0&&o->visible[e]&&own[e]>=0&&own[e]!=p&&moveArmy(e)>landing){exposed=true;break;}
    }
    if(exposed)continue;
    double score=1.45+std::min(3.8,.70*std::sqrt(double(sent))) /
                              (1+targetDist[s]*.06)
      +std::min(.8,army(dest)*.03)+(knownGeneral>=0?.35:0)
      -(s==g?.6:0)-departure;
    score=std::min(score,5.6);
    if(score>best.score)best={score,s,goal,dir,half};
   }
  }
  return best;
 }
 // Receding-horizon route search. Every node is a simple path: a garrison
 // can be collected only once, and the plan values all captures along it.
 // The search has a fixed node bound, independent of machine timing.
 Candidate routePlan(double income) {
  Candidate best;int count=0;
  int concentration=20;
  bool harvestActive=harvestValue>0&&tick>=50;
  bool contacts[8]{};
  if(harvestContact)for(int c=0;c<N;c++)if(o->visible[c]&&own[c]>=0&&own[c]!=p)contacts[own[c]]=true;
  if(harvestContact)harvestActive=harvestActive&&std::any_of(contacts,contacts+o->players,[](bool known){return known;});
  if(harvestActive)for(int e=0;e<o->players;e++)if(e!=p&&o->alive[e]&&(!harvestContact||contacts[e]))
   concentration=std::max(concentration,12+int(std::min<int64_t>(240,std::max<int64_t>(0,o->total_army[e]-o->total_land[e]))*.45));
  for(int s:sources){
   bool requiredHalf=false;if(!canDepart(s,requiredHalf))continue;
   int modes=(!requiredHalf&&usefulCityHalf(s))?2:1;
   for(int mode=0;mode<modes;mode++){
   bool half=requiredHalf||mode;
   if(moveArmy(s,half)<1)continue;
   Route r;r.cell=r.source=s;r.force=moveArmy(s)+1;r.half=half;r.path[0]=s;
   beam[count++]=r;
   }
  }
  double discount[13];discount[0]=1;for(int d=1;d<=12;d++)discount[d]=discount[d-1]*.93;
  const double troopScale=income*2.5+2;
  for(int depth=1;depth<=(harvestActive?12:10)&&count;depth++){
   int generated=0;
   std::fill(cellRank,cellRank+N,-1e30);
   for(int b=0;b<count;b++){
    const Route &r=beam[b];int u=r.cell;
    int moving=(depth==1?moveArmy(r.source,r.half):r.force-1+
       int((actionTurn(depth)%2==0&&st[u]>0&&st[u]<3)||(actionTurn(depth)%50==0&&st[u]==0)));
    if(moving<1)continue;
    for(int dir=0;dir<4;dir++){
     int v=nb[u][dir];if(v<0||blocked(v))continue;
     bool repeated=false;for(int j=0;j<depth;j++)if(r.path[j]==v){repeated=true;break;}
     if(repeated)continue;
     int guard=rememberedDefense(v);
     if(own[v]>=0&&!mine(v)&&st[v]>0&&st[v]<3)guard+=actionTurn(depth)/2-actionTurn(1)/2;
     if(!mine(v)&&moving<=guard+(ter[v]==255?1:0))continue;
     if(mine(v)&&v==g&&reserve[g]>1)continue;
     Route next=r;next.cell=v;next.steps=depth;next.path[depth]=v;
     if(depth==1){
      next.dir=dir;next.hazard=risk(u,v,moving);
      if(next.hazard>=1000)continue;
      // Earlier seats can consume the source before this preselected move.
      for(int k=0;k<4;k++){
       int e=nb[u][k];if(e<0||!o->visible[e]||own[e]<0||own[e]>=p)continue;
       if(moveArmy(e)>r.force)next.hazard+=8;
      }
     }
     // Retain the worst landing exposure throughout this candidate route;
     // visiting more squares around the same defending stack does not incur
     // that same possible loss repeatedly.
     double exposure=0;
     if(!mine(v))for(int k=0;k<4;k++){
      int e=nb[v][k];if(e<0||e==u||!o->visible[e]||own[e]<0||own[e]==p)continue;
      bool captured=false;for(int j=0;j<depth;j++)if(r.path[j]==e){captured=true;break;}
      if(!captured&&moveArmy(e)>moving-guard+2)exposure+=std::min(5.,army(e)*.025);
     }
     next.exposure=std::max(r.exposure,exposure);
     if(mine(v)){
      next.force=moving+std::max(0,army(v)-reserve[v]+1);
      if(st[v]>0&&st[v]<3)next.force+=futureGrowth(depth);
      else if(st[v]==0)next.force+=futureGrowth(depth,50);
      if(harvestActive){
       int collected=std::max(0,next.force-moving-1);
       // Merging a one-troop donor into an existing large army must not
       // receive credit for assembling that whole army again.
       int largestBefore=std::max(moving+1,next.force-moving);
       int useful=std::min({collected,moving,std::max(0,concentration-largestBefore)});
       double proximity=harvestContact?std::clamp((18.-ed[v])/8.,0.,1.):1.;
       double production=harvestIncome?std::min(1.,1.5/income):1.;
       next.gain+=discount[depth-1]*harvestValue*useful*proximity*production;
      }
     }else{
      next.force=moving-guard-(ter[v]==255?1:0);
      next.loss+=guard+(ter[v]==255?1:0);
      bool enemy=own[v]>=0&&own[v]!=p,city=st[v]==1,king=st[v]==2&&enemy;
      double reward=0;
      if(king){
       if(depth>1&&next.force<=2)continue;
       if(depth==1&&moving<=captureDefense(v))continue;
       if(depth==1&&aliveCount>2&&!captureSafe(r.source,v,moving))continue;
       reward=eliminationValue(own[v]);
      }else if(city){
       reward=15+std::min(10.,army(r.source)*.05)+(enemy?5:0);
       if(depth<=4&&next.force>=8)reward+=3+(enemy?2:0);
       if(next.force<4)reward*=.72;
       if(tick<95)reward*=.75;
       if(o->players>2&&tick>=140)reward*=1.4;
      }else{
       reward=1.8+(enemy?2.1:0)+(gd[v]<8?.3:0);
       int fresh=0,x=v%W,y=v/W;
       for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){
        int xx=x+dx,yy=y+dy;if(xx<0||xx>=W||yy<0||yy>=H)continue;
        int c=yy*W+xx;if(ter[c]!=255)continue;
        bool revealed=false;
        for(int j=1;j<depth;j++)if(std::abs(r.path[j]%W-xx)<=1&&std::abs(r.path[j]/W-yy)<=1){revealed=true;break;}
        if(!revealed)fresh++;
       }
       reward+=fresh*(tick<80?.32:.22);
       if(enemy&&moving>=20&&next.force>=8)reward+=fresh*.55;
       if(enemy&&guard>3)reward+=std::min(6.,guard*.06);
       if(ter[v]==255)reward*=.76;
      }
      next.gain+=discount[depth-1]*reward;
     }
     next.loss+=(depth>1);
     next.rank=next.gain/(.8+.80*depth+next.loss/troopScale)-next.hazard-next.exposure;
     if(next.source==oldSource)next.rank*=1.025;
     if(next.cell==oldTarget)next.rank*=1.04;
     if(next.gain>0&&next.rank>best.score){
      best={next.rank,next.source,v,next.dir,next.half};
     }
     // A small terminal value preserves gathering routes in the beam without
     // awarding their speculative gain to an executable action.
     double priority=next.rank+.08*std::sqrt(double(next.force));
     if(priority<cellRank[v]-.20)continue;
     cellRank[v]=std::max(cellRank[v],priority);
     next.rank=priority;
     if(generated<384)expanded[generated++]=next;
    }
   }
   // Twelve-step assembly uses fewer beam nodes than the ten-step base search.
   count=std::min(harvestActive?64:96,generated);
   std::partial_sort(expanded,expanded+count,expanded+generated,[](const Route&a,const Route&b){return a.rank>b.rank;});
   std::copy(expanded,expanded+count,beam);
  }
  return best;
 }
 // A rooted friendly tree turns a bounded gather into a small knapsack:
 // joining a child costs one action, independently of its number of troops.
 // Each subtree is disjoint, so no soldier is promised to two transfers.
 void gatherTree(int root,int budget){
  std::fill(treeParent,treeParent+N,-2);
  treeCount=1;treeOrder[0]=root;treeParent[root]=-1;dist[root]=0;
  for(int j=0;j<treeCount;j++){
   int u=treeOrder[j];if(dist[u]>=budget)continue;
   for(int k=0;k<4;k++){
    int v=nb[u][k];if(v<0||!mine(v)||treeParent[v]!=-2)continue;
    if(v==g&&root!=g&&reserve[g]>1)continue;
    treeParent[v]=u;dist[v]=dist[u]+1;treeOrder[treeCount++]=v;
   }
  }
  for(int j=treeCount-1;j>=0;j--){
   int u=treeOrder[j];
   int base=moveArmy(u)+1;
   for(int k=0;k<=budget;k++){gatherForce[u][k]=base;gatherSource[u][k]=-1;gatherDir[u][k]=-1;}
   for(int direction=0;direction<4;direction++){
    int c=nb[u][direction];if(c<0||treeParent[c]!=u)continue;
    for(int total=budget;total>=1;total--){
     for(int used=0;used<total;used++){
      int gain=gatherForce[c][used]-1;if(gain<=0)continue;
      int value=gatherForce[u][total-used-1]+gain;
      if(value<=gatherForce[u][total])continue;
      int source=gatherSource[c][used],dir=gatherDir[c][used];
      if(source<0){source=c;dir=(direction+2)%4;}
      // Direct arrivals occur first when they can reach the root immediately.
      if(gatherSource[u][total-used-1]>=0&&gatherSource[u][total-used-1]!=u){
       int other=gatherSource[u][total-used-1];
       if(dist[other]<dist[source]){source=other;dir=gatherDir[u][total-used-1];}
      }
      gatherForce[u][total]=value;gatherSource[u][total]=source;gatherDir[u][total]=dir;
     }
    }
   }
  }
 }
 Candidate siegePlan(double income){
  Candidate best;
  if(tick<65)return best;
  int objectives[128],objectCount=0;
  for(int c=0;c<N&&objectCount<128;c++)if(!mine(c)&&
     (st[c]==1||(st[c]==2&&own[c]>=0&&own[c]!=p)))objectives[objectCount++]=c;
  if(!objectCount)return best;
  for(int si=0;si<std::min(8,int(sources.size()));si++){
   int root=sources[si];if(moveArmy(root)<4)continue;
   bool near=false;
   for(int j=0;j<objectCount;j++)if(std::abs(root/W-objectives[j]/W)+std::abs(root%W-objectives[j]%W)<=12){near=true;break;}
   if(!near)continue;
   gatherTree(root,8);
   int gathered[9],actionSource[9],actionDir[9];
   for(int k=0;k<=8;k++){
    gathered[k]=gatherForce[root][k];actionSource[k]=gatherSource[root][k];actionDir[k]=gatherDir[root][k];
   }
   std::fill(dist,dist+N,INF);std::fill(pathLoss,pathLoss+N,INF);
   int h=0,z=0;queue[z++]=root;dist[root]=0;pathLoss[root]=0;first[root]=-1;
   while(h<z){
    int u=queue[h++];if(dist[u]>=12)continue;
    for(int k=0;k<4;k++){
     int v=nb[u][k];if(v<0||blocked(v)||v==root)continue;
     int d=dist[u]+1;if(dist[v]<d)continue;
     int loss=pathLoss[u]+1+(mine(v)?0:rememberedDefense(v)+(ter[v]==255?1:0));
     if(dist[v]==INF){dist[v]=d;queue[z++]=v;}
     if(loss<pathLoss[v]){pathLoss[v]=loss;first[v]=u==root?k:first[u];}
    }
   }
   for(int j=0;j<objectCount;j++){
    int target=objectives[j],d=dist[target];if(d>12||first[target]<0)continue;
    bool enemy=own[target]>=0,king=st[target]==2&&enemy;
    double reward=king?eliminationValue(own[target]):
                      18+std::min(10.,army(root)*.05)+(enemy?7:0);
    if(!king&&tick<95)reward*=.75;
    for(int k=1;k<=8;k++){
     int source=actionSource[k],direction=actionDir[k];if(source<0||moveArmy(source)<=0)continue;
     int available=gathered[k]-1;
     if(reserve[root]>1){if((gathered[k]+1)/2<reserve[root])continue;available=gathered[k]/2;}
     int expectedGrowth=enemy?actionTurn(k+d)/2-actionTurn(1)/2:0;
     int remaining=available-pathLoss[target]+1-expectedGrowth;
     if(remaining<(king?3:7))continue;
     // Only reward assembly when the current stack cannot already do the job.
     if(moveArmy(root)-pathLoss[target]+1-expectedGrowth>=(king?3:7))break;
     int dest=nb[source][direction];double danger=risk(source,dest,moveArmy(source));
     if(danger>=1000)continue;
     bool exposed=false;
     for(int z=0;z<4;z++){
      int e=nb[source][z];if(e>=0&&o->visible[e]&&own[e]>=0&&own[e]<p&&moveArmy(e)>moveArmy(source)+1)exposed=true;
     }
     if(exposed)continue;
     double score=reward/(d+k+.8+(pathLoss[target]+k)/(income*2.5+2));
     if(king)score+=6./(1+.07*(d+k));
     if(target==oldTarget)score*=1.08;
     score-=danger;
     if(score>best.score)best={score,source,target,direction,false};
     break;
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
  if(defense.score<=22){Candidate forecast=forecastDefend();if(forecast.score>defense.score)defense=forecast;}
  if(o->players<=6){Candidate merged=mergerShield(defense);if(merged.source>=0)defense=merged;}
  // A threatened general takes precedence unless an immediate elimination exists.
  if(defense.score>22){
   oldTarget=defense.target;
   oldSource=nb[defense.source][defense.dir];
   lastTarget=oldSource;
   return arena_move(defense.source,defense.dir,defense.half);
  }

  double income=.5*(std::max<int64_t>(0,o->total_cities[p])+1)+.02*o->total_land[p];
  sources.clear();keep.clear();border.clear();for(int i=0;i<N;i++)if(mine(i)&&canMoveNow(i))sources.push_back(i);
  std::sort(sources.begin(),sources.end(),[&](int x,int y){double ax=army(x)+(x==oldSource?5:0),ay=army(y)+(y==oldSource?5:0);return ax>ay;});
  if(sources.size()>16){
   // Preserve the twelve strongest mobile groups. Four further slots keep
   // useful border armies from being crowded out by productive rear cities.
   if(o->players<=6){
    keep.assign(sources.begin(),sources.begin()+12);
    int scout=-1,scoutGain=0,scoutDistance=INF;
    for(int j=12;j<int(sources.size());j++){
     int source=sources[j],key=ed[source],visionGain=0,capturable=0;
     bool half=false;
     if(canDepart(source,half)){
      int moving=moveArmy(source,half);
      for(int k=0;k<4;k++){
       int v=nb[source][k];
       if(v>=0&&!blocked(v)&&!mine(v)&&moving>rememberedDefense(v)){
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
    // Search ten strong mobile groups and four active frontier groups on
    // mature crowded maps.
    keep.assign(sources.begin(),sources.begin()+10);
    for(int j=10;j<int(sources.size());j++){
     int source=sources[j],key=ed[source];
     if(key>=INF){
      // Before enemy contact, use new vision and immediately capturable
      // neighboring terrain as the exploration-front substitute.
      key=INF-unknown[source]*2;
      for(int k=0;k<4;k++){
       int v=nb[source][k];
       if(v>=0&&!blocked(v)&&!mine(v)&&moveArmy(source)>rememberedDefense(v))key--;
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
   bool requiredHalf=false;if(!canDepart(s,requiredHalf))continue;
   int modes=(!requiredHalf&&usefulCityHalf(s))?2:1;
   for(int mode=0;mode<modes;mode++){
   bool half=requiredHalf||mode;
   int moving=moveArmy(s,half);if(moving<=0)continue;
   // These tactical adjustments depend on the first step, not on the final
   // campaign target. Store hazards individually to preserve subtraction order
   // and therefore the original floating-point tie behavior exactly.
   double stepRisk[4]{};bool cityCounter[4]{};
   for(int direction=0;direction<4;direction++){
    int dest=nb[s][direction];if(dest<0)continue;
    stepRisk[direction]=risk(s,dest,moving);
    if(own[dest]!=p){
     int remain=moving-rememberedDefense(dest);
     for(int k=0;k<4;k++){
      int e=nb[dest][k];
      if(e>=0&&e!=s&&o->visible[e]&&own[e]>=0&&own[e]!=p){
       if(o->players==2&&st[dest]==1&&remain>0){
        int attack=moveArmy(e),hold=remain;
        if(own[e]<p){
         attack+=int((actionTurn(2)%2==0&&st[e]>0&&st[e]<3)||(actionTurn(2)%50==0&&st[e]==0));
         hold+=int(actionTurn(2)%2==0);
        }
        if(attack>hold+2)cityCounter[direction]=true;
       }
      }
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
      reward=eliminationValue(own[v]);
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
      if(enemy&&o->visible[v]&&ndef<=4&&moving>=18&&force[v]>=8)
       reward+=std::min(1.4,fogGain[v]*.18);
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
    if(city&&o->players>2&&tick>=140)reward*=1.4;
    double lost=pathLoss[v];
    double den=d+.8+lost/(income*2.5+2);
    double score=reward/den;
    if(king)score+=6.0/(1+d*.07);
    if(immediateKing)score+=100000;
    if(v==oldTarget)score*=(o->players==2&&ed[s]<=8?1.05:1.13);
    if(s==oldSource)score*=1.025;
    if(city&&tick<95)score*=.75;
    int direction=first[v];
    int next=nb[s][direction];
    if(!mine(next)&&moving>rememberedDefense(next)&&!king)
     score=std::max(score,deploymentValue(s,next,moving,d-1));
    if(cityCounter[direction])score*=.35;
    // A lethal departure is infeasible; a numerical king bonus must not
    // cancel its risk penalty. Preserve immediate last-opponent captures.
    if(o->players<=6&&stepRisk[direction]>=1000&&!(alive<=2&&immediateKing))continue;
    score-=stepRisk[direction];
    if(!king)score-=pathExposure[v];
    if(score>best.score)best={score,s,v,first[v],half};
    }
   }
  }
  }
  if(aliveCount==2)for(int j=0;j<std::min(1,int(sources.size()));j++){
   Candidate detour=weightedCampaign(sources[j],income);if(detour.score>best.score)best=detour;
   if(usefulCityHalf(sources[j])){
    Candidate split=weightedCampaign(sources[j],income,true);if(split.score>best.score)best=split;
   }
  }
  if(best.score<22){Candidate route=routePlan(income);if(route.score>best.score)best=route;}
  if(o->players>2&&tick>=250&&best.score<10){Candidate siege=siegePlan(income);if(siege.score>best.score)best=siege;}
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
  if(best.source>=0&&best.target>=0&&!mine(best.target)&&best.score<22&&!protectInvestment){
   int goal=best.target;
   bfs(goal,targetDist,false,true);
   for(int s:sources){if((saveCity&&s==g)||ed[s]<=1)continue;bool half=false;if(!canDepart(s,half))continue;int n=moveArmy(s,half);if(n<3)continue;
    for(int k=0;k<4;k++){int v=nb[s][k];if(v<0||!mine(v)||targetDist[v]>=targetDist[s]||targetDist[s]>=INF)continue;
      // Gathering is useful when it shortens the time to an attainable city or assault.
      double score=deploymentValue(s,v,n,targetDist[v]);
      score-=risk(s,v,n);
      if(score>best.score)best={score,s,goal,k,half};
    }
   }
  }
  if(best.score<8){Candidate rally=fieldRally(best);if(aliveCount>2)rally.score*=.90;if(rally.score>best.score)best=rally;}
  if(aliveCount==2&&best.score<4.25){
   Candidate relief=cityRelief();if(relief.score>best.score)best=relief;
  }
  if(best.source<0||best.score<0.10)return 0;
  oldTarget=best.target;oldSource=nb[best.source][best.dir];lastTarget=oldSource;
  return arena_move(best.source,best.dir,best.half);
 }
};
}
extern "C" void *arena_create(int player,std::uint64_t seed,const char *params) { return new Policy(player,seed,params); }
extern "C" std::int64_t arena_act(void *instance,const ArenaObservation *observation) {
 auto *policy=static_cast<Policy*>(instance);
 auto action=policy->act(observation);policy->rememberAction(observation,action);return action;
}
extern "C" void arena_destroy(void *instance) { delete static_cast<Policy*>(instance); }
