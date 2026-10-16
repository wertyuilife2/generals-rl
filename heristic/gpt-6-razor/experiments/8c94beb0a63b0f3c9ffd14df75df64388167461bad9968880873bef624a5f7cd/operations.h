// Strategic estimates use only this seat's observation history. They feed
// executable programs into the common action scheduler in controller.h.
int assetHeldSince[MAXN]{},assetLostAt[MAXN]{},assetFlips[MAXN]{};
int assetHeldTotal[MAXN]{},assetCaptures[MAXN]{},assetLastOwner[MAXN]{};
bool assetHistoryReady=false;
int assetGuard[MAXN]{},assetETA[MAXN]{},zoneThreat[MAXN]{},zoneETA[MAXN]{};
int assetRelief[MAXN][25]{};
struct CityReplyOrder {short s=-1,t=-1;bool half=false;};
struct CityReply {int count=0;CityReplyOrder orders[24];};
int cityReplyIndex[MAXN];
CityReply cityReplies[20][25];
int priorCityArmy[MAXN]{},priorCityTick[MAXN]{},priorCityOwner[MAXN]{};
bool priorCityThreat[MAXN]{};
double cityDefenseHits[8]{},cityDefenseCases[8]{};
double intel[MAXN]{},intelGain[MAXN]{},routeCost[MAXN];
double kingInformation[MAXN]{},informationGain[MAXN]{},informationPrice=0;
int opParent[MAXN],opSteps[MAXN],opForce[MAXN];
bool opHalf[MAXN];
bool conversionMode=false;
int conversionSince=-1000;
double observedColumnFraction[8]{};
int columnObservationTick=-1;
int opponentAssembly[MAXN]{};

// Forecast a short, executable concentration by the opponent. The required
// field army should answer what a nearby column can become, not just the
// number currently displayed on its square. Shared transport edges and
// donors are counted once, under a six-order collection budget.
void buildOpponentAssembly(){
 std::fill(opponentAssembly,opponentAssembly+N,0);
 if(tick<120)return;
 int starts[12],count=0;
 for(int c=0;c<N;c++)if(o->visible[c]&&own[c]>=0&&own[c]!=p&&st[c]!=2&&ready(c)>=12){
  int j=count;if(j==12){j=11;if(ready(c)<=ready(starts[j]))continue;}else count++;
  while(j>0&&ready(c)>ready(starts[j-1])){if(j<12)starts[j]=starts[j-1];j--;}
  starts[j]=c;
 }
 for(int ri=0;ri<count;ri++){
  int root=starts[ri],enemy=own[root];
  std::fill(scheduleParent,scheduleParent+N,-2);
  int nodes=1;scheduleOrder[0]=root;scheduleParent[root]=-1;dist[root]=0;
  for(int j=0;j<nodes;j++){
   int u=scheduleOrder[j];if(dist[u]>=6)continue;
   for(int k=0;k<4;k++){int v=nb[u][k];
    if(v<0||blocked(v)||own[v]!=enemy||scheduleParent[v]!=-2||nodes>=64)continue;
    scheduleParent[v]=u;dist[v]=dist[u]+1;scheduleOrder[nodes++]=v;
   }
  }
  for(int j=nodes-1;j>=0;j--){
   int u=scheduleOrder[j];schedule[u][0]=std::max(1,rememberedDefense(u));
   if(u!=root&&st[u]==2)schedule[u][0]=schedule[u][0]/2+1;
   for(int b=1;b<=6;b++)schedule[u][b]=-INF;
   for(int k=0;k<4;k++){int v=nb[u][k];if(v<0||scheduleParent[v]!=u)continue;
    for(int b=6;b>=1;b--)for(int use=0;use<b;use++)
     if(schedule[u][b-use-1]>0&&schedule[v][use]>1)
      schedule[u][b]=std::max(schedule[u][b],schedule[u][b-use-1]+schedule[v][use]-1);
   }
   for(int b=1;b<=6;b++)schedule[u][b]=std::max(schedule[u][b],schedule[u][b-1]);
  }
  opponentAssembly[root]=schedule[root][6];
 }
}

void observeColumns(){
 int elapsed=columnObservationTick<0?0:std::max(0,tick-columnObservationTick);
 columnObservationTick=tick;
 for(int e=0;e<o->players;e++)observedColumnFraction[e]=
  std::max(.35,observedColumnFraction[e]-.0003*elapsed);
 for(int c=0;c<N;c++)if(o->visible[c]&&own[c]>=0&&own[c]!=p&&st[c]!=2&&army(c)>=16){
  int e=own[c];double mobile=std::max<int64_t>(1,o->total_army[e]-o->total_land[e]);
  double fraction=std::min(.95,std::max(0,army(c)-1)/mobile);
  observedColumnFraction[e]=std::max(observedColumnFraction[e],fraction);
 }
}

int assemblyRequirement(int root)const {
 bool near[8]{};int need=24;
 for(int c=0;c<N;c++)if(own[c]>=0&&own[c]!=p&&o->alive[own[c]]&&tick-seen[c]<=80){
  int distance=std::abs(cellX[c]-cellX[root])+std::abs(cellY[c]-cellY[root]);
  if(distance<=14)near[own[c]]=true;
  if(o->visible[c]&&distance<=10)need=std::max(need,std::max(army(c),opponentAssembly[c])+12);
 }
 for(int e=0;e<o->players;e++)if(near[e]){
  double mobile=std::max<int64_t>(0,o->total_army[e]-o->total_land[e]);
  need=std::max(need,4+int(mobile*std::max(.35,observedColumnFraction[e])));
 }
 return std::min(need,int(std::max<int64_t>(0,o->total_army[p]-o->total_land[p])*.65));
}

void strategicMode(){
 int enemyArmy=0;bool unknownKing=false;
 for(int e=0;e<o->players;e++)if(e!=p&&o->alive[e]){
  enemyArmy=std::max(enemyArmy,int(o->total_army[e]));
  if(enemyGeneral[e]<0)unknownKing=true;
 }
 if(!unknownKing||tick<260||o->total_cities[p]<2){conversionMode=false;return;}
 double ratio=double(o->total_army[p]+20)/(enemyArmy+20);
 if(!conversionMode&&ratio>=1.12){conversionMode=true;conversionSince=tick;}
 if(conversionMode&&ratio<.95&&tick-conversionSince>20)conversionMode=false;
}

void assetHistory(){
 for(int c=0;c<N;c++){
  int now=mine(c)?p:own[c];
  if(!assetHistoryReady){assetLastOwner[c]=now;if(now==p&&st[c]==1)assetHeldSince[c]=tick;}
  else if(st[c]==1){
   if(now==p&&assetLastOwner[c]!=p){assetHeldSince[c]=tick;assetCaptures[c]++;}
   if(now!=p&&assetLastOwner[c]==p){
    int held=std::max(1,tick-assetHeldSince[c]);assetHeldTotal[c]+=held;
    assetLostAt[c]=tick;if(held<30)assetFlips[c]=std::min(12,assetFlips[c]+2);
   }
   if(tick%100==0&&assetFlips[c]>0)assetFlips[c]--;
  }
  assetLastOwner[c]=now;
 }
 assetHistoryReady=true;
}

void observeCityResponses(){
 for(int c=0;c<N;c++)if(st[c]==1&&o->visible[c]&&own[c]>=0&&own[c]!=p){
  int enemy=own[c];
  if(priorCityTick[c]==tick-1&&priorCityOwner[c]==enemy&&priorCityThreat[c]){
   cityDefenseCases[enemy]+=1.;
   if(army(c)-priorCityArmy[c]-grows(c,tick)>=4)cityDefenseHits[enemy]+=1.;
   if(cityDefenseCases[enemy]>40){cityDefenseCases[enemy]*=.8;cityDefenseHits[enemy]*=.8;}
  }
  int attacking=0,donor=0;
  for(int k=0;k<4;k++){
   int v=nb[c][k];if(v<0||!o->visible[v])continue;
   if(mine(v))attacking=std::max(attacking,moveArmy(v));
   else if(own[v]==enemy)donor=std::max(donor,moveArmy(v));
  }
  priorCityThreat[c]=attacking>ready(c)&&donor>=5;
  priorCityArmy[c]=army(c);priorCityOwner[c]=enemy;priorCityTick[c]=tick;
 }
}

void pressureMap(){
 std::fill(zoneThreat,zoneThreat+N,0);std::fill(zoneETA,zoneETA+N,INF);
 int starts[24],count=0;
 for(int c=0;c<N;c++)if(own[c]>=0&&own[c]!=p&&o->alive[own[c]]&&
    (o->visible[c]||tick-seen[c]<=8)&&rememberedDefense(c)>3){
  int n=rememberedDefense(c),j=count;
  if(j==24){j=23;if(n<=rememberedDefense(starts[j]))continue;}else count++;
  while(j>0&&n>rememberedDefense(starts[j-1])){if(j<24)starts[j]=starts[j-1];j--;}
  starts[j]=c;
 }
 for(int si=0;si<count;si++){
  int source=starts[si],enemy=own[source];
  std::fill(opSteps,opSteps+N,INF);std::fill(opForce,opForce+N,0);
  int h=0,z=1;queue[0]=source;opSteps[source]=0;opForce[source]=rememberedDefense(source);
  while(h<z){int u=queue[h++],d=opSteps[u],power=opForce[u];
   // Keep the pre-combat arrival at our asset, not only its surviving force.
   if(u!=source&&power>zoneThreat[u]){zoneThreat[u]=power;zoneETA[u]=d;}
   else if(u!=source&&power>=zoneThreat[u]*.8)zoneETA[u]=std::min(zoneETA[u],d);
   if(d>=6)continue;
   int available=power;
   if(u!=source){
    if(mine(u))available-=ready(u);
    else if(own[u]==enemy&&o->visible[u])available+=std::max(0,ready(u)-1);
    else if(own[u]==-2)available-=rememberedDefense(u);
   }
   if(available<=2)continue;
   for(int k=0;k<4;k++){int v=nb[u][k];if(v<0||blocked(v)||v==source||opSteps[v]<d+1)continue;
    if(opSteps[v]==INF){opSteps[v]=d+1;queue[z++]=v;}
    opForce[v]=std::max(opForce[v],available-1);
   }
  }
 }
 for(int c=0;c<N;c++)if(st[c]==1){
  int combined[8]{},count[8]{};
  for(int k=0;k<enemyCount[c];k++){int e=enemyNeighbors[c][k];combined[own[e]]+=moveArmy(e);count[own[e]]++;}
  for(int e=0;e<o->players;e++)if(count[e]>=2&&combined[e]>zoneThreat[c]){
   zoneThreat[c]=combined[e];zoneETA[c]=count[e];
  }
 }
 // Retention is a portfolio decision: several individually defensible
 // cities must not each claim the same field army or consume the entire army.
 int candidates[MAXN],countCities=0;
 double priority[MAXN];
 int mobile=std::max<int64_t>(0,o->total_army[p]-o->total_land[p]);
 int guardBudget=std::max(12,int(mobile*.25));
 for(int c=0;c<N;c++){
  assetGuard[c]=1;assetETA[c]=zoneETA[c];
  if(st[c]!=1||!mine(c)||zoneETA[c]>3)continue;
  int need=std::max(1,zoneThreat[c]-futureGrowth(std::max(0,zoneETA[c]))+2);
  // Reserve actual troops at the asset. Future reinforcements are separate
  // executable tasks, never counted as already present at several cities.
  if(need>ready(c)||need>guardBudget)continue;
  priority[c]=(40.+(gd[c]<=6?15.:0)+std::min(20,army(c)-need)*.15)/(need+8.);
  assetGuard[c]=need;candidates[countCities++]=c;
 }
 std::sort(candidates,candidates+countCities,[&](int x,int y){return priority[x]>priority[y];});
 for(int j=0;j<countCities;j++){
  int c=candidates[j],need=assetGuard[c];
  if(need<=guardBudget){reserve[c]=need;guardBudget-=need;}
  else assetGuard[c]=1;
 }

 for(int c=0;c<N;c++)if(mine(c))routeDelta[c]=c==g&&reserve[g]>1?0:std::max(0,army(c)-reserve[c]+1);
}

double assetUtility(int c,int remaining,int arrival)const {
 // Repeated quick losses are evidence that a nominal capture does not buy
 // the promised income. Until the counterattack can be met, price this as
 // transit through ordinary land rather than another productive conquest.
 if(assetFlips[c]>=2&&tick-assetLostAt[c]<40&&remaining<zoneThreat[c]+8)
  return own[c]>=0&&own[c]!=p?3.9:1.8;
 int local=0;
 for(int k=0;k<enemyCount[c];k++)local=std::max(local,moveArmy(enemyNeighbors[c][k]));
 double holding=80.;
 if(zoneThreat[c]>remaining+3&&zoneETA[c]<=6)holding=8.+6.*zoneETA[c];
 if(local>remaining+3)holding=std::min(holding,8.);
 else if(remaining>local+12)holding=std::min(90.,holding+16.);
 int pressure=zoneThreat[c],eta=zoneETA[c];
 if(pressure>remaining+3&&eta<=3)holding=std::min(holding,double(std::max(2,eta-arrival+2)));
 if(assetFlips[c]>0&&remaining<pressure+20)holding=std::min(holding,24./(1+.4*assetFlips[c]));
 int exits=0;for(int k=0;k<4;k++){int v=nb[c][k];if(v>=0&&!blocked(v)&&st[v]!=1)exits++;}
 double value=(5.+holding*.29+exits*.55)/(1+.012*arrival);

 if(own[c]>=0&&own[c]!=p)value+=holding*.29/(1+.012*arrival)/std::max(1,aliveCount-1);
 if(remaining<4)value*=.70;
 if(assetFlips[c]>0&&remaining<pressure+20)value-=std::min(10.,1.5*assetFlips[c]);
 if(own[c]>=0&&own[c]!=p){
  int enemy=own[c];double response=(2.+cityDefenseHits[enemy])/(8.+cityDefenseCases[enemy]);
  int extra=assetRelief[c][std::clamp(arrival,0,12)];
  value*=std::clamp(double(std::max(1,remaining))/(4.+response*extra),.15,1.);
 }
 return value;
}

// A neutral-city purchase consumes our army without removing an opponent's
// army. Price the resulting position, including the field column which can
// return to the city, rather than counting the purchase itself as success.
// Public mobile army supplies a cautious prior when the front is visible;
// it is not assigned to a hidden square or treated as an observed attacker.
double capitalSecurity(const Program &q)const {
 if(q.goal>=0&&st[q.goal]==2&&own[q.goal]>=0&&own[q.goal]!=p)return 1.;
 bool purchase=false;
 for(int j=q.cursor;j<q.count;j++){
  int c=q.orders[j].t;
  if(c>=0&&st[c]==1&&own[c]<0&&ed[c]<=6){purchase=true;break;}
 }
 if(!purchase)return 1.;
 int index[MAXN],cells[81],owners[81],troops[81],count=0,assets[40],purchaseCost[40],assetCount=0;
 std::fill(index,index+N,-1);
 for(int j=q.cursor;j<q.count;j++)for(int c:{q.orders[j].s,q.orders[j].t}){
  if(c<0||c>=N||blocked(c))return .25;
  if(index[c]>=0)continue;
  int i=count++;index[c]=i;cells[i]=c;owners[i]=own[c];
  troops[i]=mine(c)?ready(c):rememberedDefense(c);
 }
 for(int j=q.cursor;j<q.count;j++){
  int step=j-q.cursor+1;
  if(step>1)for(int i=0;i<count;i++)if(owners[i]>=0)troops[i]+=grows(cells[i],actionTurn(step));
  auto x=q.orders[j];int s=index[x.s],t=index[x.t];
  if(owners[s]!=p)return .25;
  int sent=x.half?troops[s]/2:troops[s]-1;
  if(sent<1||troops[s]-sent<reserve[x.s])return .25;
  troops[s]-=sent;
  if(owners[t]==p)troops[t]+=sent;
  else{
   if(sent<=troops[t])return .25;
   if(st[x.t]==1&&owners[t]<0&&ed[x.t]<=6){assets[assetCount]=x.t;purchaseCost[assetCount++]=troops[t];}
   troops[t]=sent-troops[t];owners[t]=p;
  }
 }
 auto ownerAt=[&](int c){return index[c]<0?own[c]:owners[index[c]];};
 auto forceAt=[&](int c){return index[c]<0?ready(c):troops[index[c]];};
 int distance[MAXN],work[MAXN];double security=1.;
 for(int ai=0;ai<assetCount;ai++){
  int city=assets[ai];bool front[8]{};int visibleColumn=0;
  std::fill(distance,distance+N,INF);distance[city]=0;int head=0,tail=1;work[0]=city;
  while(head<tail){int c=work[head++],d=distance[c];
   if(own[c]>=0&&own[c]!=p&&o->alive[own[c]]&&tick-seen[c]<=8){
    front[own[c]]=true;if(o->visible[c])visibleColumn=std::max(visibleColumn,army(c));
   }
   if(d>=6)continue;
   for(int k=0;k<4;k++){int v=nb[c][k];if(v<0||blocked(v)||distance[v]<INF)continue;
    distance[v]=d+1;work[tail++]=v;}
  }
  double need=visibleColumn;
  for(int e=0;e<o->players;e++)if(front[e])
   need=std::max(need,std::max<int64_t>(0,o->total_army[e]-o->total_land[e])*
                      std::max(.35,observedColumnFraction[e]));
  if(need<16)continue;
  // The best single returning column is a feasible support option. Do not
  // add several donors whose individual distances conceal multiple orders.
  int support=0,horizon=std::clamp(ed[city],2,6);
  std::fill(distance,distance+N,INF);distance[city]=0;head=0;tail=1;work[0]=city;
  while(head<tail){int c=work[head++],d=distance[c];
   if(c!=city&&ownerAt(c)==p){
    int n=forceAt(c),sent=reserve[c]>1?n/2:n-1;
    if(n-sent>=reserve[c])support=std::max(support,sent-d+1);
   }
   if(d>=horizon)continue;
   for(int k=0;k<4;k++){int v=nb[c][k];if(v<0||ownerAt(v)!=p||distance[v]<INF)continue;
    // A protected intermediate city cannot promise an unrestricted full
    // transfer of this column on its way back to the new asset.
    if(c!=city&&reserve[c]>1)continue;
    distance[v]=d+1;work[tail++]=v;}
  }
  int held=ownerAt(city)==p?forceAt(city):0;
  // A purchase that consumes at least as much army as the whole remaining
  // garrison and returnable support, while leaving that force below the
  // estimated nearby counter-column, is not a funded occupation. Reject the
  // investment even when no optional scouting candidate has been generated.
  if(held+support<need&&purchaseCost[ai]>=held+support)return 0.;
  security=std::min(security,.15+.85*std::min(1.,(held+support)/(need+4.)));
 }
 return security;
}

// Price the actual orders a visible defender has before occupation. Separate
// donors share their common transport edges and are never counted twice.
void buildCityRelief(){
 std::memset(assetRelief,0,N*sizeof(assetRelief[0]));
 std::fill(cityReplyIndex,cityReplyIndex+N,-1);
 int roots[20],count=0;double scores[12];
 for(int c=0;c<N;c++)if(st[c]==1&&o->visible[c]&&own[c]>=0&&own[c]!=p){
  int nearby=0;for(int k=0;k<4;k++){int u=nb[c][k];if(u>=0&&mine(u))nearby=std::max(nearby,ready(u));}
  double value=nearby/(rememberedDefense(c)+10.)+1./(rememberedDefense(c)+1.);
  int j=count;if(j==12){j=11;if(value<=scores[j])continue;}else count++;
  while(j>0&&value>scores[j-1]){if(j<12){roots[j]=roots[j-1];scores[j]=scores[j-1];}j--;}
  roots[j]=c;scores[j]=value;
 }
 // A general's relief must share the attack ledger too. Otherwise an army
 // killed on the approach can be charged again as a future defender.
 for(int e=0;e<o->players;e++)if(e!=p&&o->alive[e]){
  int c=enemyGeneral[e];if(c>=0&&o->visible[c]&&own[c]==e)roots[count++]=c;
 }
 for(int ri=0;ri<count;ri++){
  int root=roots[ri],enemy=own[root];
  int budgetLimit=st[root]==2?24:12,depthLimit=st[root]==2?16:8,nodeLimit=st[root]==2?128:64;
  cityReplyIndex[root]=ri;
  std::fill(scheduleParent,scheduleParent+N,-2);
  int nodes=1;scheduleOrder[0]=root;scheduleParent[root]=-1;dist[root]=0;
  for(int j=0;j<nodes;j++){
   int u=scheduleOrder[j];if(dist[u]>=depthLimit)continue;
   for(int k=0;k<4;k++){
    int v=nb[u][k];if(v<0||blocked(v)||own[v]!=enemy||scheduleParent[v]!=-2||nodes>=nodeLimit)continue;
    scheduleParent[v]=u;dist[v]=dist[u]+1;scheduleOrder[nodes++]=v;
   }
  }
  for(int j=nodes-1;j>=0;j--){
   int u=scheduleOrder[j];schedule[u][0]=std::max(1,rememberedDefense(u));
   if(u!=root&&st[u]==2)schedule[u][0]=schedule[u][0]/2+1;
   for(int b=1;b<=budgetLimit;b++)schedule[u][b]=-INF;
   for(int k=0;k<4;k++){
    for(int b=0;b<=budgetLimit;b++)schedulePick[u][k][b]=-1;
    int v=nb[u][k];if(v<0||scheduleParent[v]!=u)continue;
    for(int b=budgetLimit;b>=1;b--)for(int use=0;use<b;use++)
     if(schedule[u][b-use-1]>0&&schedule[v][use]>1){
      int value=schedule[u][b-use-1]+schedule[v][use]-1;
      if(value>schedule[u][b]){schedule[u][b]=value;schedulePick[u][k][b]=use;}
     }
   }
  }
  int bestBudget[25]{};
  for(int d=1;d<=budgetLimit;d++)bestBudget[d]=schedule[root][d]>schedule[root][bestBudget[d-1]]?d:bestBudget[d-1];
  for(int d=0;d<=budgetLimit;d++){
   auto &reply=cityReplies[ri][d];reply.count=0;
   extractCityReply(root,bestBudget[d],reply);
  }
  for(int d=1;d<=budgetLimit;d++){
   int budget=std::max(0,d-int(enemy>p));
   assetRelief[root][d]=std::max(0,schedule[root][bestBudget[budget]]-rememberedDefense(root));
  }
 }
}

void extractCityReply(int u,int budget,CityReply &reply){
 int use[4];
 for(int k=3;k>=0;k--){use[k]=schedulePick[u][k][budget];if(use[k]>=0)budget-=use[k]+1;}
 for(int k=0;k<4;k++)if(use[k]>=0){
  int child=nb[u][k];extractCityReply(child,use[k],reply);
  if(reply.count<24)reply.orders[reply.count++]={short(child),short(u),st[child]==2};
 }
}

// Test a concrete observed/history-derived reinforcement response against
// the whole remaining occupation program. Both players move on one ledger
// in seat order: a donor killed en route cannot also reinforce the city, and
// a later seat can recapture immediately after our apparent final capture.
double occupationReplyConfidence(const Program &q,int extraDefense=0)const {
 int goal=q.goal,steps=q.count-q.cursor;
 if(goal<0||(st[goal]!=1&&st[goal]!=2)||own[goal]<0||own[goal]==p||steps<1)return 1.;
 int ri=cityReplyIndex[goal];if(ri<0||ri>=20)return 1.;
 // A city can be recaptured in the same round; a defeated general cannot
 // receive a later seat's final reinforcement after elimination.
 int budget=steps-int(st[goal]==2&&own[goal]>p);
 const auto &reply=cityReplies[ri][std::clamp(budget,0,st[goal]==2?24:12)];
 if(!reply.count&&!extraDefense)return 1.;
 int index[MAXN],cells[129],owners[129],troops[129],count=0;
 std::fill(index,index+N,-1);
 auto add=[&](int c){if(index[c]>=0)return;int i=count++;index[c]=i;cells[i]=c;
  owners[i]=own[c];troops[i]=mine(c)?ready(c):rememberedDefense(c);};
 for(int j=q.cursor;j<q.count;j++){add(q.orders[j].s);add(q.orders[j].t);}
 for(int j=0;j<reply.count;j++){add(reply.orders[j].s);add(reply.orders[j].t);}
 add(goal);troops[index[goal]]+=extraDefense;
 int enemy=own[goal];bool eliminated=false;double arrivalRatio=.2;
 auto move=[&](int source,int target,bool half,int player){
  int s=index[source],t=index[target];if(owners[s]!=player)return false;
  int sent=half?troops[s]/2:troops[s]-1;if(sent<1)return false;
  if(target==goal&&player==p&&owners[t]!=p)arrivalRatio=sent/(troops[t]+3.);
  troops[s]-=sent;
  if(owners[t]==player)troops[t]+=sent;
  else if(sent>troops[t]){
   if(st[target]==2&&owners[t]==enemy&&player==p)eliminated=true;
   troops[t]=sent-troops[t];owners[t]=player;
  }else troops[t]-=sent;
  return true;
 };
 for(int step=0;step<steps;step++){
  if(step)for(int i=0;i<count;i++)if(owners[i]>=0)troops[i]+=grows(cells[i],actionTurn(step+1));
  auto respond=[&](){if(step<reply.count){auto x=reply.orders[step];move(x.s,x.t,x.half,enemy);}};
  if(enemy<p)respond();
  auto x=q.orders[q.cursor+step];if(!move(x.s,x.t,x.half,p))return .2;
  if(eliminated)return 1.;
  if(enemy>p)respond();
 }
 return owners[index[goal]]==p?1.:std::clamp(arrivalRatio,.2,.99);
}

bool cityReplyHolds(const Program &q)const {
 if(q.goal<0||st[q.goal]!=1)return true;
 return occupationReplyConfidence(q)>=1.;
}

double kingProgramConfidence(const Program &q,int remaining,int depth)const {
 int goal=q.goal,steps=q.count-q.cursor;
 if(goal<0)return 1.;
 double old=kingConfidence(goal,remaining,depth);
 if(goal<0||st[goal]!=2||!o->visible[goal]||own[goal]<0||own[goal]==p||
    cityReplyIndex[goal]<0||steps<1||steps>24)return old;
 // Preserve the old model's longer single-donor warning if the bounded
 // response tree did not cover it. Only represented troops share the ledger.
 int extra=std::max(0,kingRelief[own[goal]][steps]-assetRelief[goal][steps]);
 return occupationReplyConfidence(q,extra);
}

void intelligenceMap(){
 std::fill(intel,intel+N,0.);std::fill(intelGain,intelGain+N,0.);
 std::fill(kingInformation,kingInformation+N,0.);std::fill(informationGain,informationGain+N,0.);
 informationPrice=0;
 double ex[8]{},ey[8]{},weight[8]{},spread[8]{};
 for(int c=0;c<N;c++)if(own[c]>=0&&own[c]!=p&&o->alive[own[c]]){
  int e=own[c];double w=st[c]==1?3.:1.;ex[e]+=cellX[c]*w;ey[e]+=cellY[c]*w;weight[e]+=w;
 }
 for(int e=0;e<o->players;e++)if(e!=p&&o->alive[e]&&enemyGeneral[e]<0){
  spread[e]=4.+std::sqrt(double(o->total_land[e]))*.5;
  if(weight[e]){
   ex[e]/=weight[e];ey[e]/=weight[e];
   double dx=ex[e]-cellX[g],dy=ey[e]-cellY[g],norm=std::sqrt(dx*dx+dy*dy);
   double behind=std::min(6.,std::sqrt(double(o->total_land[e]))*.4);
   if(norm>0){ex[e]=std::clamp(ex[e]+behind*dx/norm,0.,double(W-1));ey[e]=std::clamp(ey[e]+behind*dy/norm,0.,double(H-1));}
  }else {ex[e]=W-1-cellX[g];ey[e]=H-1-cellY[g];}
 }
 for(int c=0;c<N;c++){
  if(ter[c]==1)continue;
  if(seen[c]<0&&ter[c]!=2){
   for(int e=0;e<o->players;e++)if(e!=p&&o->alive[e]&&enemyGeneral[e]<0){
    double dx=cellX[c]-ex[e],dy=cellY[c]-ey[e];
    double away=std::clamp((gd[c]-6.)/12.,.10,1.);
    double prior=weight[e]?1.8*std::exp(-(dx*dx+dy*dy)/(2*spread[e]*spread[e])):1.;
    intel[c]=std::max(intel[c],away*prior);
   }
  }
  kingInformation[c]=intel[c];
  if(!o->visible[c]&&st[c]==1)intel[c]+=.3*std::min(1.,(tick-seen[c])/40.);
  if(!o->visible[c]&&ter[c]==2)intel[c]+=.18;
 }
 double total=0,prize=0;int strongest=0;
 for(int c=0;c<N;c++)total+=kingInformation[c];
 for(int e=0;e<o->players;e++)if(e!=p&&o->alive[e]){
  strongest=std::max(strongest,int(o->total_army[e]));
  if(enemyGeneral[e]<0)prize=std::max(prize,eliminationValue(e));
 }
 double ratio=double(o->total_army[p]+20)/(strongest+20.);
 if(total>0)informationPrice=prize*std::clamp(.25+.35*(ratio-1.),.12,.65)/total;
 for(int c=0;c<N;c++)if(!blocked(c)){
  for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){
   int x=cellX[c]+dx,y=cellY[c]+dy;if(x>=0&&x<W&&y>=0&&y<H){
    intelGain[c]+=intel[y*W+x];informationGain[c]+=informationPrice*kingInformation[y*W+x];
   }
  }
 }
}


void taskRoutes(int source,int initial,bool recon=false,bool friendly=false){
 std::fill(routeCost,routeCost+N,1e20);std::fill(opParent,opParent+N,-1);
 std::fill(opSteps,opSteps+N,INF);std::fill(opForce,opForce+N,0);
 using Node=std::pair<double,int>;std::priority_queue<Node,std::vector<Node>,std::greater<Node>> heap;
 routeCost[source]=0;opSteps[source]=0;opForce[source]=initial;heap.push({0,source});
 int expanded=0;
 while(!heap.empty()&&expanded<N*2){
  auto [cost,u]=heap.top();heap.pop();if(cost>routeCost[u]+1e-6)continue;expanded++;
  if(opSteps[u]>=40)continue;
  for(int k=0;k<4;k++){
   int v=nb[u][k];if(v<0||blocked(v)||v==source||(friendly&&!mine(v)))continue;
   int guard=mine(v)?0:rememberedDefense(v),available=opForce[u];bool half=false;
   int step=opSteps[u]+1,period=st[v]>0&&st[v]<3?2:50;
   int growth=std::max(0,actionTurn(step)/period-actionTurn(1)/period);
   if(u!=source){
    int prepared=opForce[u]+grows(u,actionTurn(step));half=reserve[u]>1;
    available=half?prepared/2:prepared-1;if(prepared-available<reserve[u])continue;
   }
   if(mine(v))available+=ready(v)+growth;
   else available-=guard+(own[v]>=0?growth:0);
   if(available<2&&!friendly)continue;
   int mobileThreat=zoneThreat[v];
   double riskCost=std::max(0,mobileThreat-available)* (recon?.14:.04);
   if(recon&&zoneETA[v]<=2&&mobileThreat>available+5)riskCost+=8.;
   double next=cost+1.+guard/(recon?5.:18.)+riskCost+(ter[v]==255?.20:0.);
   if(next+1e-6>=routeCost[v])continue;
   routeCost[v]=next;opParent[v]=u;opHalf[v]=half;opSteps[v]=opSteps[u]+1;opForce[v]=std::max(1,available);heap.push({next,v});
  }
 }
}

bool permitted(const Order &x,bool emergency=false)const {
 if(!legalOrder(x,emergency))return false;
 int moving=moveArmy(x.s,x.half);
 if(!emergency&&risk(x.s,x.t,moving)>=1000)return false;
 if(!emergency&&st[x.s]==1&&reserve[x.s]>1&&ready(x.s)-moving<reserve[x.s]){
  // Removing the threatening adjacent army can itself retain control.
  if(mine(x.t)||zoneThreat[x.s]>moving-rememberedDefense(x.t)+ready(x.s)-moving)return false;
 }
 double after=dangerAfter(&x);
 return emergency||after<=baselineDanger+1||after<2;
}

int routeFirst(int source,int goal)const {
 if(goal<0||goal==source||opParent[goal]<0)return -1;
 int c=goal,steps=0;
 while(opParent[c]!=source){c=opParent[c];if(c<0||++steps>N)return -1;}
 return c;
}

void reconnaissancePrograms(){
 if(!conversionMode||baselineDanger>0||o->total_land[p]<40)return;
 int roots=0;
 for(int source:sources){
  bool half=false;if(!canDepart(source,half))continue;
  // Reconnaissance leases part of a column; the remainder keeps the
  // front occupied and remains available to answer a counterattack.
  if(usefulCityHalf(source)||ready(source)>=40)half=true;
  int initial=moveArmy(source,half);if(initial<12)continue;
  if(++roots>5)break;
  taskRoutes(source,initial,true);
  double best=-1e30;int goal=-1;
  for(int c=0;c<N;c++)if(!mine(c)&&opSteps[c]<24&&opSteps[c]>0&&intelGain[c]>2&&opForce[c]>5){
   double revealed=0;int steps=0;
   for(int u=c;u>=0&&u!=source&&steps++<32;u=opParent[u])revealed+=intelGain[u];
   double reward=2.+std::min(14.,revealed*.45)+std::min(6.,std::sqrt(double(initial))*.3);
   double value=reward/(1.5+.35*routeCost[c]);
   value=5.5*(-std::expm1(-value/6.));
   if(source==rallyCell&&tick-rallyTick<=20)value+=1.5;
   int next=routeFirst(source,c);if(next<0)continue;
   Order x{source,next,half};if(!permitted(x))continue;
   if(value>best){best=value;goal=c;}
  }
  if(goal<0)continue;
  Program q;q.root=source;q.goal=goal;q.born=tick;q.kind=10;q.value=best;
  int path[40],n=0;for(int c=goal;c!=source&&c>=0&&n<40;c=opParent[c])path[n++]=c;
  int u=source;for(int j=n-1;j>=0;j--){q.orders[q.count++]={u,path[j],j==n-1?half:opHalf[path[j]]};u=path[j];}
  offer(q);
 }
}

int mobileAssemblyCredit(const Program &q,int root,int finalForce,int need)const {
 int cells[41],families[41],mass[41]{},count=0;
 cells[0]=root;for(int j=0;j<q.count;j++)cells[j+1]=q.orders[j].s;
 for(int j=0;j<=q.count;j++){
  int c=cells[j],family=currentFlow(c);bool split=family!=0;if(!family)family=-c-1;
  int slot=0;while(slot<count&&families[slot]!=family)slot++;
  if(slot==count)families[count++]=family;
  bool half=false;
  int usable=canDepart(c,half)?moveArmy(c,half):0;
  mass[slot]+=split?std::max(0,ready(c)-1):usable;
 }
 int largest=1;for(int j=0;j<count;j++)largest=std::max(largest,mass[j]+1);
 return std::min(std::max(0,finalForce-largest),std::max(0,need-largest));
}

void mobilizationPrograms(){
 bool knownTarget=false;for(int e=0;e<o->players;e++)if(e!=p&&o->alive[e]&&enemyGeneral[e]>=0)knownTarget=true;
 // The first enemy column often arrives before either side has two cities.
 // A nearby observed front, rather than an economic lead, warrants assembly.
 if(tick<120||o->total_land[p]<25||baselineDanger>0)return;
 int roots[6],count=0;double priorities[6];
 for(int c=0;c<N;c++)if(mine(c)&&st[c]==0&&ed[c]<=10){
  if(zoneETA[c]<=4&&zoneThreat[c]>=ready(c)-2)continue;
  double nearby=ready(c);
  for(int k=0;k<4;k++){
   int v=nb[c][k];if(v<0||!mine(v))continue;
   bool half=false;if(canDepart(v,half))nearby+=moveArmy(v,half)*.7;
   for(int j=0;j<4;j++){int u=nb[v][j];if(u>=0&&u!=c&&mine(u)&&canDepart(u,half))nearby+=moveArmy(u,half)*.15;}
  }
  double priority=nearby/(3.+.5*ed[c]);int slot=count;
  if(count==6){slot=5;if(priority<=priorities[slot])continue;}else count++;
  while(slot>0&&priority>priorities[slot-1]){if(slot<6){roots[slot]=roots[slot-1];priorities[slot]=priorities[slot-1];}slot--;}
  roots[slot]=c;priorities[slot]=priority;
 }
 for(int ri=0;ri<count;ri++){
  if(!planningTimeLeft())break;
  int root=roots[ri],need=assemblyRequirement(root);
  if(ready(root)>=need&&!knownTarget)continue;
  int budget=tick<500?12:24;
  scheduleTree(root,budget);Program best;
  // The attack leg pays movement and combat losses without recollecting the
  // same donors already used by the collection tree.
  std::fill(opSteps,opSteps+N,INF);std::fill(opForce,opForce+N,INF);std::fill(opParent,opParent+N,-1);
  int head=0,tail=1;queue[0]=root;opSteps[root]=0;opForce[root]=0;
  while(head<tail){int u=queue[head++];if(opSteps[u]>=24)continue;
   for(int j=0;j<4;j++){int v=nb[u][j];if(v<0||blocked(v)||v==root||opSteps[v]<opSteps[u]+1||(mine(v)&&reserve[v]>1))continue;
    if(opSteps[v]==INF){opSteps[v]=opSteps[u]+1;queue[tail++]=v;}
    int loss=opForce[u]+1+(mine(v)?0:rememberedDefense(v));
    if(loss<opForce[v]){opForce[v]=loss;opParent[v]=u;}
   }
  }
  for(int k=1;k<=budget;k++)if(schedule[root][k]>ready(root)+4){
   Program q;q.root=root;q.goal=root;q.kind=11;q.born=tick;extractSchedule(root,k,q);
   int useful=mobileAssemblyCredit(q,root,schedule[root][k],need);
   q.value=.70*useful/(k+2.)/(1.+.08*ed[root]);
   q.value=9.*(-std::expm1(-q.value/9.));
   if(q.value>best.value&&q.count&&permitted(q.orders[0]))best=q;
   for(int e=0;e<o->players;e++)if(e!=p&&o->alive[e]&&enemyGeneral[e]>=0){
    int goal=enemyGeneral[e],d=opSteps[goal];if(d>=INF||k+d>40)continue;
    int remaining=schedule[root][k]-opForce[goal]-(k+d)/2;
    int relief=kingRelief[e][std::min(32,k+d)];
    if(remaining<relief+5)continue;
    Program assault=q;assault.kind=15;assault.goal=goal;
    assault.value=eliminationValue(e)/(k+d+1.+opForce[goal]/12.)+6./(1.+.07*(k+d));
    int path[25],n=0;for(int c=goal;c!=root&&c>=0&&n<25;c=opParent[c])path[n++]=c;
    int u=root;for(int j=n-1;j>=0;j--){assault.orders[assault.count++]={u,path[j],false};u=path[j];}
    if(assault.value>best.value&&permitted(assault.orders[0]))best=assault;
   }
  }
  offer(best);
 }
}


void assetProtectionPrograms(){
 if(tick<100||baselineDanger>0)return;
 int considered=0;
 for(int c=0;c<N;c++)if(mine(c)&&st[c]==1&&zoneETA[c]<=6&&zoneETA[c]>1){
  int need=zoneThreat[c]-futureGrowth(zoneETA[c])+2;
  if(need<=ready(c)||need>o->total_army[p]*.35)continue;
  if(++considered>6)break;
  int budget=std::min(6,zoneETA[c]-1);scheduleTree(c,budget);
  for(int k=1;k<=budget;k++)if(schedule[c][k]>=need){
   Program q;q.kind=12;q.goal=c;q.root=c;q.born=tick;
   q.value=4.+8./(k+1.);extractSchedule(c,k,q);
   if(q.count&&permitted(q.orders[0]))offer(q);
   break;
  }
 }
}

// Removing a mobile enemy army changes what the whole front can do. Its
// value cannot be bounded by the prize of the single ordinary tile it holds.
void fieldEngagementPrograms(){
 for(int s=0;s<N;s++)if(mine(s)&&ready(s)>12)for(int k=0;k<enemyCount[s];k++){
  int t=enemyNeighbors[s][k],guard=rememberedDefense(t);
  if(guard<12||st[t]==2)continue;
  for(int half=0;half<2;half++){
   Order x{s,t,bool(half)};int moving=moveArmy(s,half),left=moving-guard;
   if(left<4||!legalOrder(x)||risk(s,t,moving)>=1000)continue;
   int counter=0;
   for(int j=0;j<enemyCount[t];j++){int c=enemyNeighbors[t][j];if(c!=s)counter=std::max(counter,moveArmy(c));}
   if(counter>left+2)continue;
   double value=3.+.035*guard+.012*std::min(left,guard);
   if(st[t]==1)value+=2.;
   value-=risk(s,t,moving);
   offer(oneStep({value,s,t,direction(s,t),bool(half)},13));
  }
 }
}

bool extractionSafe(const Program &plan)const {
 if(plan.cursor>=plan.count||plan.count>3||plan.root!=g)return false;
 int cells[8],forces[8],count=0;
 auto add=[&](int c){for(int j=0;j<count;j++)if(cells[j]==c)return j;
  int j=count++;cells[j]=c;forces[j]=ready(c);return j;};
 int home=add(g);
 for(int j=plan.cursor;j<plan.count;j++){add(plan.orders[j].s);add(plan.orders[j].t);}
 for(int d=1;d<=8;d++){
  int before=forces[home],j=plan.cursor+d-1;
  if(j<plan.count){
   auto x=plan.orders[j];if(!mine(x.s)||!mine(x.t))return false;
   int s=add(x.s),t=add(x.t),sent=x.half?forces[s]/2:forces[s]-1;
   if(sent<1)return false;
   forces[s]-=sent;forces[t]+=sent;
   // The return donor must survive until the scheduled return. A troop
   // already captured by a visible army cannot be promised to home.
   if(x.s!=g&&zoneETA[x.s]<=d&&zoneThreat[x.s]>forces[s])return false;
   if(x.t!=g&&zoneETA[x.t]<=d+1&&zoneThreat[x.t]>forces[t])return false;
  }
  if(homeIncoming[d]+2>std::min(before,forces[home]))return false;
 }
 return forces[home]>=reserve[g];
}

void extractionPrograms(){
 if(tick<150||ready(g)<70||reserve[g]<=1||ready(g)>=2*reserve[g])return;
 int n=ready(g);if(n-n/4-2<reserve[g])return;
 for(int c=0;c<N;c++)if(own[c]>=0&&own[c]!=p&&!o->visible[c]&&tick-seen[c]<=8&&
    gd[c]-(tick-seen[c])<=3&&rememberedDefense(c)>n/2)return;
 for(int k=0;k<4;k++){
  int u=nb[g][k];if(u<0||!mine(u)||st[u]!=0||army(u)>n/5)continue;
  for(int d=0;d<4;d++){
   int v=nb[u][d];if(v<0||v==g||!mine(v)||st[v]!=0)continue;
   Program q;q.kind=14;q.root=g;q.goal=v;q.born=tick;q.count=3;
   q.orders[0]={g,u,true};q.orders[1]={u,v,true};q.orders[2]={u,g,false};
   q.value=1.+.22*(n/4)/(3.+.25*std::min(15,ed[v]));
   if(extractionSafe(q))offer(q);
  }
 }
}
