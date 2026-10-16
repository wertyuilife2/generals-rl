// Razor v8: persistent operations, reconnaissance, and defensible production.
// All persistent knowledge is derived from this seat's observations. The
// tactical safety and observation layer is adapted from our kestrel series.
#include "arena_policy.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

namespace {
constexpr int MAXN=4096, INF=1000000;
// Which squares of a target's 3x3 vision are also revealed by a path cell
// at each relative offset in [-2,2]^2. Immutable geometry, shared read-only.
constexpr unsigned VISION_OVERLAP[25]={1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};
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
 unsigned halfMask=0;
 short path[13]{};
};
struct Policy {
 std::chrono::steady_clock::time_point planningStart=std::chrono::steady_clock::now();
 bool planningTimeLeft()const {
  return std::chrono::steady_clock::now()-planningStart<std::chrono::microseconds(1800);
 }
 int p,W=0,H=0,N=0,g=-1,tick=0,oldTarget=-1,oldSource=-1,lastTarget=-1,aliveCount=0;
 uint64_t rng;
 double harvestValue=0;
 bool harvestContact=false,harvestIncome=false;
 double networkValue=1,raidValue=1;
 double capitalHorizon=80;
 int homeDistance[MAXN],component[MAXN],componentArmy[MAXN],componentLand[MAXN];
 double networkBonus[MAXN];
 int lastHomeContact=-1000,raidNeed=0;
 bool growthBeforeMove=true;
 int growthEvidence=0,receiptSource=-1,receiptTarget=-1,receiptTick=-1;
 int receiptArmy=0,receiptTargetArmy=0,receiptOwner=-1;
 bool receiptHalf=false;

 int ter[MAXN],st[MAXN],own[MAXN],seen[MAXN],a[MAXN],lastOwn[MAXN];
 int enemyCount[MAXN],enemyNeighbors[MAXN][4],cellX[MAXN],cellY[MAXN];
 unsigned unknownMask[MAXN];
 int nb[MAXN][4],gd[MAXN],fd[MAXN],ed[MAXN],unknown[MAXN],fogGain[MAXN],reserve[MAXN],targetDist[MAXN];
 int dist[MAXN],force[MAXN],pathLoss[MAXN],first[MAXN],parent[MAXN],queue[MAXN];
 double threat[MAXN],pathExposure[MAXN];
 bool pathHalf[MAXN]{};
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
  networkValue=std::clamp(parameter(params,"\"network_value\"",1),0.0,2.0);
  raidValue=std::clamp(parameter(params,"\"raid_value\"",1),0.0,2.0);
  capitalHorizon=std::clamp(parameter(params,"\"capital_horizon\"",80),0.0,640.0);
  std::fill(ter,ter+MAXN,255);std::fill(st,st+MAXN,255);std::fill(own,own+MAXN,-3);
  std::fill(a,a+MAXN,0);std::fill(seen,seen+MAXN,-10000);std::fill(lastOwn,lastOwn+MAXN,-3);
  std::fill(enemyGeneral,enemyGeneral+8,-1);
  sources.reserve(MAXN);keep.reserve(MAXN);border.reserve(MAXN);
  campaignHeap.reserve(MAXN*4+4);
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
  if(W==0){W=o->width;H=o->height;N=W*H;for(int i=0;i<N;i++){int x=i%W,y=i/W;cellX[i]=x;cellY[i]=y;nb[i][0]=y?i-W:-1;nb[i][1]=x+1<W?i+1:-1;nb[i][2]=y+1<H?i+W:-1;nb[i][3]=x?i-1:-1;}}
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
  // Every search uses the same visible enemy adjacency. Cache it once
  // in direction order instead of rescanning four observation arrays for
  // every candidate path and source. This does not change candidate scores.
  for(int c=0;c<N;c++){
   enemyCount[c]=0;
   for(int k=0;k<4;k++){
    int e=nb[c][k];
    if(e>=0&&o->visible[e]&&own[e]>=0&&own[e]!=p)
     enemyNeighbors[c][enemyCount[c]++]=e;
   }
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
   if(mapChanged){unknown[i]=0;unknownMask[i]=0;}
   for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){
    int xx=x+dx,yy=y+dy;
    if(xx>=0&&xx<W&&yy>=0&&yy<H){int c=yy*W+xx;if(!o->visible[c]&&!blocked(c))fogGain[i]++;if(mapChanged&&ter[c]==255){unknown[i]++;unknownMask[i]|=1u<<((dy+1)*3+dx+1);}}
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
  // A compact opponent can have little visible force while preparing a raid.
  // The estimate uses public totals, recent home contact and our own force
  // distribution; it never assigns an unseen stack to a specific square.
  raidNeed=0;
  if(ed[g]<=6)lastHomeContact=tick;
  if(raidValue>0&&aliveCount==2&&tick>=70&&tick-lastHomeContact<=30){
   int largestField=0;
   for(int c=0;c<N;c++)if(mine(c)&&c!=g)largestField=std::max(largestField,army(c));
   for(int e=0;e<o->players;e++)if(e!=p&&o->alive[e]&&
       o->total_land[e]*3<o->total_land[p]*2&&o->total_army[e]*2>=o->total_army[p]){
    // An imminent global growth pulse can fund the raid before our reserves
    // can be rebuilt. Forecast that public, guaranteed income as well.
    int pulses=(tick+6)/50-tick/50;
    int field=int(std::min<int64_t>(300,std::max<int64_t>(0,
              o->total_army[e]-o->total_land[e]+pulses*o->total_land[e])));
    if(field>=25&&largestField*2<field)
     raidNeed=std::max(raidNeed,int(raidValue*std::min(field*.40,double(o->total_army[p])*.30)));
   }
   reserve[g]=std::max(reserve[g],raidNeed);
  }
  updateFlows();
  homeInfrastructure();

  for(int i=0;i<N;i++){
   if(mine(i)){
    routeDelta[i]=std::max(0,army(i)-reserve[i]+1);
    if(i==g&&reserve[i]>1)routeDelta[i]=0;
   }else routeDelta[i]=-defenseCache[i]-(ter[i]==255?1:0);
  }
 }
 // Evaluate transport capacity only when home has actually lost access to
 // most of the domain. Ordinary fragments and distant shortcuts receive no
 // windfall that could distract from expansion or a viable king attack.
 void homeInfrastructure(){
  std::fill(networkBonus,networkBonus+N,0.);
  if(networkValue<=0||o->total_land[p]<20)return;
  bfs(g,homeDistance,true);
  int homeLand=0,homeArmy=0;
  for(int c=0;c<N;c++)if(mine(c)&&homeDistance[c]<INF){homeLand++;homeArmy+=army(c);}
  if(homeLand*2>=o->total_land[p]||homeArmy*2>=o->total_army[p])return;
  std::fill(component,component+N,-1);
  for(int c=0;c<N;c++)if(homeDistance[c]<INF)component[c]=0;
  componentLand[0]=homeLand;componentArmy[0]=homeArmy;int count=1;
  for(int s=0;s<N;s++)if(mine(s)&&component[s]<0){
   int h=0,z=0;queue[z++]=s;component[s]=count;
   componentLand[count]=componentArmy[count]=0;
   while(h<z){
    int u=queue[h++];componentLand[count]++;componentArmy[count]+=army(u);
    for(int k=0;k<4;k++){
     int v=nb[u][k];if(v<0||!mine(v)||component[v]>=0)continue;
     component[v]=count;queue[z++]=v;
    }
   }
   count++;
  }
  for(int c=0;c<N;c++)if(!mine(c)&&!blocked(c)){
   int groups[4],n=0;bool home=false;
   for(int k=0;k<4;k++){
    int v=nb[c][k];if(v<0||!mine(v))continue;
    int id=component[v];home|=id==0;bool known=false;
    for(int j=0;j<n;j++)known|=groups[j]==id;
    if(!known)groups[n++]=id;
   }
   if(!home||n<2)continue;
   int land=0,army=0;
   for(int j=0;j<n;j++)if(groups[j]>0){land+=componentLand[groups[j]];army+=componentArmy[groups[j]];}
   if(land<8||army<15)continue;
   networkBonus[c]=networkValue*std::min(20.,std::sqrt(double(army)));
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
 // Price a counterattack along the whole proposed path. Charging it only
 // on step one lets a receding-horizon planner postpone the same hazard
 // forever by walking away and then back. A captured path cell cannot also
 // supply a hostile counterattack in that plan.
 double landingExposure(int from,int dest,int remaining,const int *parents,const int *depths)const {
  if(mine(dest))return 0;
  double exposure=0;
  for(int k=0;k<enemyCount[dest];k++){
   int e=enemyNeighbors[dest][k];
   if(e==from||a[e]<=remaining+2)continue;
   bool captured=false;
   if(depths[e]<depths[from])
    for(int c=from;c>=0;c=parents[c])if(c==e){captured=true;break;}
   if(!captured)exposure+=std::min(5.,a[e]*.025);
  }
  return exposure;
 }
 // Fast bounded breadth-first campaigns, keeping the strongest shortest route.
 void search(int s,int initial,int depth){
  std::fill(pathHalf,pathHalf+N,false);
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
    if(f>force[v]||(f==force[v]&&spent<pathLoss[v])){force[v]=f;pathLoss[v]=spent;first[v]=(u==s?k:first[u]);parent[v]=u;pathExposure[v]=std::max(pathExposure[u],landingExposure(u,v,f,parent,dist));}
   }
  }
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
  // Schedule a complete reinforcement tree before the attack deadline.
  // Greedy nearest-donor transfers repeatedly traverse emptied corridors;
  // their individual ETAs do not prove that all promised troops can arrive.
  // A disjoint tree accounts for every merge and move in the action budget.
  bool scheduled=false;
  if(deadline>0){
   int budget=std::min(12,deadline);
   gatherTree(g,budget);
   int rootGrowth=moveArmy(g)+1-army(g);
   for(int used=1;used<=budget;used++){
    int source=gatherSource[g][used],direction=gatherDir[g][used];
    if(source<0||direction<0)continue;
    int delivered=gatherForce[g][used]-rootGrowth-army(g);
    if(delivered<deficit)continue;
    int dest=nb[source][direction],moving=moveArmy(source);
    double departure=risk(source,dest,moving);
    if(departure>=1000)continue;
    best={24+deficit*2.0/(used+.5)+urgency*2-departure,source,g,direction,false};
    scheduled=true;break;
   }
  }
  // When no complete rescue is feasible, retain the existing best-effort
  // reinforcement and interception options. Do not fabricate a saved king.
  for(int s=0;s<N;s++)if(!scheduled&&mine(s)&&s!=g&&canMoveNow(s)&&fd[s]<=deadline){
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
 // Receding-horizon route search. Every node is a simple path: a garrison
 // can be collected only once, and the plan values all captures along it.
 // The search has a fixed node bound, independent of machine timing.
 Candidate routePlan(double income) {
  Candidate best;Program bestProgram;int count=0;
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
   if(!planningTimeLeft())break;
   int generated=0;
   std::fill(cellRank,cellRank+N,-1e30);
   for(int b=0;b<count;b++){
    const Route &r=beam[b];int u=r.cell;
    bool departingHalf=depth==1?r.half:reserve[u]>1;
    int available=r.force+(depth>1?grows(u,actionTurn(depth)):0);
    int moving=depth==1?moveArmy(r.source,r.half):(departingHalf?available/2:available-1);
    if(available-moving<reserve[u])continue;
    if(moving<1)continue;
    for(int dir=0;dir<4;dir++){
     int v=nb[u][dir];if(v<0||blocked(v))continue;
     bool repeated=false;for(int j=0;j<depth;j++)if(r.path[j]==v){repeated=true;break;}
     if(repeated)continue;
     int guard=rememberedDefense(v);
     if(own[v]>=0&&!mine(v)&&st[v]>0&&st[v]<3)guard+=actionTurn(depth)/2-actionTurn(1)/2;
     else if(own[v]>=0&&!mine(v)&&st[v]==0)guard+=actionTurn(depth)/50-actionTurn(1)/50;
     if(!mine(v)&&moving<=guard+(ter[v]==255?1:0))continue;
     if(mine(v)&&v==g&&reserve[g]>1)continue;
     Route next=r;next.cell=v;next.steps=depth;next.path[depth]=v;
     if(departingHalf)next.halfMask|=1u<<(depth-1);
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
     if(!mine(v))for(int k=0;k<enemyCount[v];k++){
      int e=enemyNeighbors[v][k];if(e==u)continue;
      bool captured=false;for(int j=0;j<depth;j++)if(r.path[j]==e){captured=true;break;}
      if(!captured&&moveArmy(e)>moving-guard+2)exposure+=std::min(5.,army(e)*.025);
     }
     next.exposure=std::max(r.exposure,exposure);
     if(mine(v)){
      next.force=moving+army(v);
      if(st[v]>0&&st[v]<3)next.force+=futureGrowth(depth);
      else if(st[v]==0)next.force+=futureGrowth(depth,50);
      if(harvestActive){
       int deployable=reserve[v]>1?((next.force+1)/2>=reserve[v]?next.force/2:0):next.force-1;
       int collected=std::max(0,deployable-moving);
       // Merging a one-troop donor into an existing large army must not
       // receive credit for assembling that whole army again.
       int alone=reserve[v]>1?((ready(v)+1)/2>=reserve[v]?ready(v)/2:0):ready(v)-1;
       int largestBefore=std::max(moving+1,alone+1);
       int useful=std::min({collected,moving,std::max(0,concentration-largestBefore)});
       for(int j=0;j<depth;j++)if(sameFlow(r.path[j],v)){useful=0;break;}
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
       reward=eliminationValue(own[v])*kingConfidence(v,next.force,depth);
      }else if(city){
       reward=cityUtility(v,next.force,depth);
      }else{
       reward=1.8+(enemy?2.1:0)+(gd[v]<8?.3:0);
       unsigned freshMask=unknownMask[v];
       for(int j=1;j<depth&&freshMask;j++){
        int c=r.path[j],dx=cellX[c]-cellX[v],dy=cellY[c]-cellY[v];
        if(dx>=-2&&dx<=2&&dy>=-2&&dy<=2)
         freshMask&=~VISION_OVERLAP[(dy+2)*5+dx+2];
       }
       int fresh=__builtin_popcount(freshMask);
       reward+=fresh*(tick<80?.32:.22);
       if(enemy&&moving>=20&&next.force>=8)reward+=fresh*.55;
       if(enemy&&guard>3)reward+=std::min(6.,guard*.06);
       if(ter[v]==255)reward*=.76;
      }
      next.gain+=discount[depth-1]*(reward+(king?0:networkBonus[v]));
     }
     next.loss+=(depth>1);
     if(depth>1)next.loss+=.3*std::max(0,available-moving-(mine(u)?army(u):1));
     next.rank=next.gain/(.8+.80*depth+next.loss/troopScale)-next.hazard-next.exposure;
     if(next.source==oldSource)next.rank*=1.025;
     if(next.cell==oldTarget)next.rank*=1.04;
     if(next.gain>0&&next.rank>best.score){
      best={next.rank,next.source,v,next.dir,next.half};
      bestProgram=Program();bestProgram.value=next.rank;bestProgram.goal=v;
      bestProgram.root=next.source;bestProgram.born=tick;bestProgram.kind=1;
      for(int j=0;j<depth;j++)bestProgram.orders[bestProgram.count++]={next.path[j],next.path[j+1],bool(next.halfMask&(1u<<j))};
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
  offer(bestProgram);
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
 #include "planning.h"
 #include "operations.h"
#include "warfare.h"
#include "tactics.h"
#include "strategy.h"
#include "forecast.h"
#include "controller.h"
 int64_t act(const ArenaObservation *obs){return decideV8(obs);}

};
}
extern "C" void *arena_create(int player,std::uint64_t seed,const char *params) { return new Policy(player,seed,params); }
extern "C" std::int64_t arena_act(void *instance,const ArenaObservation *observation) {
 auto *policy=static_cast<Policy*>(instance);
 auto action=policy->act(observation);policy->rememberAction(observation,action);return action;
}
extern "C" void arena_destroy(void *instance) { delete static_cast<Policy*>(instance); }
