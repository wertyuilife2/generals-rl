// Price reconnaissance as a complete, executable investment. The collection
// and route share one troop ledger; a donor cannot be collected twice. Every
// order costs one turn, and only newly revealed general-location mass pays.
int expeditionDepth[MAXN],expeditionParent[MAXN],expeditionIndex[MAXN];
double expeditionCost[MAXN];
unsigned char expeditionVision[MAXN];
double expeditionCredit=0;
int expeditionCreditTick=0;

void updateExpeditionCredit(){
 expeditionCredit=std::min(28.,expeditionCredit+.20*std::max(0,tick-expeditionCreditTick));
 expeditionCreditTick=tick;
}

double expeditionValue(const Program &q){
 if(q.cursor>=q.count)return -1e30;
 int cells[81],owner[81],troops[81],count=0;
 std::fill(expeditionIndex,expeditionIndex+N,-1);
 std::fill(expeditionVision,expeditionVision+N,0);
 for(int j=q.cursor;j<q.count;j++)for(int c:{q.orders[j].s,q.orders[j].t}){
  if(c<0||c>=N||blocked(c))return -1e30;
  if(expeditionIndex[c]>=0)continue;
  int i=count++;expeditionIndex[c]=i;cells[i]=c;owner[i]=own[c];
  troops[i]=mine(c)?ready(c):rememberedDefense(c);
 }
 double information=0,land=0,spent=0;int remaining=0;
 for(int j=q.cursor;j<q.count;j++){
  int step=j-q.cursor+1;
  if(step>1)for(int i=0;i<count;i++)if(owner[i]>=0)troops[i]+=grows(cells[i],actionTurn(step));
  auto x=q.orders[j];int a=expeditionIndex[x.s],b=expeditionIndex[x.t];
  if(owner[a]!=p||direction(x.s,x.t)<0)return -1e30;
  int sent=x.half?troops[a]/2:troops[a]-1;
  if(sent<1||troops[a]-sent<reserve[x.s])return -1e30;
  troops[a]-=sent;
  if(owner[b]==p)troops[b]+=sent;
  else{
   if(sent<=troops[b])return -1e30;
   remaining=sent-troops[b];
   // This operation seeks information around the hostile column. It cannot
   // promise to pass through a square that the observed column controls.
   if(zoneETA[x.t]<=2&&zoneThreat[x.t]>remaining+3)return -1e30;
   land+=owner[b]>=0?3.9:1.8;spent+=troops[b];
   owner[b]=p;troops[b]=remaining;
  }
  spent+=1;remaining=troops[b];
  for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){
   int xx=cellX[x.t]+dx,yy=cellY[x.t]+dy;if(xx<0||xx>=W||yy<0||yy>=H)continue;
   int c=yy*W+xx;if(expeditionVision[c])continue;
   expeditionVision[c]=1;information+=kingInformation[c]*std::min(1.,remaining/12.);
  }
 }
 if(information<=0||remaining<4)return -1e30;
 double income=.5*(std::max<int64_t>(0,o->total_cities[p])+1)+.02*o->total_land[p];
 return (land+informationPrice*information)/(q.count-q.cursor+1.+spent/(income*2.5+2.));
}

void expeditionPrograms(){
 if(tick<120||o->total_land[p]<25||baselineDanger>1||informationPrice<=0)return;
 int roots[4],count=0;double priorities[4];
 for(int c=0;c<N;c++)if(mine(c)&&ready(c)>2){
  bool half=false;if(!canDepart(c,half))continue;
  int mobile=moveArmy(c,half);
  for(int k=0;k<4;k++){int v=nb[c][k];if(v>=0&&mine(v)&&canDepart(v,half))mobile+=moveArmy(v,half)/2;}
  double value=std::min(160,mobile)/(2.+.10*std::min(30,ed[c]));
  if(zoneETA[c]<=2&&zoneThreat[c]>=ready(c))continue;
  int j=count;if(j==4){j=3;if(value<=priorities[j])continue;}else count++;
  while(j>0&&value>priorities[j-1]){if(j<4){roots[j]=roots[j-1];priorities[j]=priorities[j-1];}j--;}
  roots[j]=c;priorities[j]=value;
 }
 double income=.5*(std::max<int64_t>(0,o->total_cities[p])+1)+.02*o->total_land[p];
 for(int ri=0;ri<count;ri++){
  if(!planningTimeLeft())break;
  int root=roots[ri];bool requiredHalf=false;canDepart(root,requiredHalf);
  scheduleTree(root,12);
  std::fill(expeditionCost,expeditionCost+N,1e30);
  std::fill(expeditionParent,expeditionParent+N,-1);
  std::fill(expeditionDepth,expeditionDepth+N,INF);
  using Entry=std::pair<double,int>;auto &heap=campaignHeap;heap.clear();
  auto push=[&](Entry e){heap.push_back(e);std::push_heap(heap.begin(),heap.end(),std::greater<Entry>());};
  expeditionCost[root]=0;expeditionDepth[root]=0;push({0,root});
  int expanded=0;
  while(!heap.empty()&&expanded<N*2){
   std::pop_heap(heap.begin(),heap.end(),std::greater<Entry>());auto [cost,u]=heap.back();heap.pop_back();
   if(cost>expeditionCost[u]+1e-8)continue;expanded++;
   if(expeditionDepth[u]>=24)continue;
   for(int k=0;k<4;k++){
    int v=nb[u][k];if(v<0||v==root||blocked(v)||(mine(v)&&reserve[v]>1))continue;
    int guard=mine(v)?0:rememberedDefense(v);
    double exposure=zoneETA[v]<=2?std::max(0,zoneThreat[v]-24)*.15:0.;
    double next=cost+1.+guard/(income*2.5+2.)+exposure;
    if(next>=expeditionCost[v])continue;
    expeditionCost[v]=next;expeditionParent[v]=u;expeditionDepth[v]=expeditionDepth[u]+1;push({next,v});
   }
  }
  int goals[6],goalCount=0;double goalPriority[6];
  for(int c=0;c<N;c++)if(informationGain[c]>0&&expeditionDepth[c]>0&&expeditionDepth[c]<=24){
   double value=informationGain[c]/(1.+expeditionCost[c]);
   int j=goalCount;if(j==6){j=5;if(value<=goalPriority[j])continue;}else goalCount++;
   while(j>0&&value>goalPriority[j-1]){if(j<6){goals[j]=goals[j-1];goalPriority[j]=goalPriority[j-1];}j--;}
   goals[j]=c;goalPriority[j]=value;
  }
  Program best;
  for(int k=0;k<=12;k++){
   if(!planningTimeLeft())break;
   if(schedule[root][k]<1)continue;
   Program collection;if(k)extractSchedule(root,k,collection);
   for(int gi=0;gi<goalCount;gi++){
    int goal=goals[gi],path[25],n=0;
    for(int u=goal;u!=root&&u>=0&&n<25;u=expeditionParent[u])path[n++]=u;
    if(!n||n>24)continue;
    for(int mode=int(requiredHalf);mode<=1;mode++){
     Program q=collection;q.kind=16;q.root=root;q.goal=goal;q.born=tick;
     int u=root;for(int j=n-1;j>=0;j--){q.orders[q.count++]={u,path[j],j==n-1&&bool(mode)};u=path[j];}
     q.value=expeditionValue(q);
     // If a half sortie obtains the same information and territory, retain
     // the other half for defense or a second operation. Full is still
     // available when the route actually requires its additional troops.
     if(q.value>best.value+1e-9||
        (mode&&q.value>=best.value-1e-9&&q.value>-1e29)){
      if(permitted(q.orders[0]))best=q;
     }
    }
   }
  }
  offer(best);
 }
}

// A bounded local adversarial search for an observed general. The defender
// may move its garrison, combine donors, attack a source, or wait. The second
// attack is chosen after observing that response, rather than assuming the
// defender leaves its troops in place. No enemy policy is inspected.
struct AssaultState {int owner[13],army[13];};
struct AssaultOrder {int s=-1,t=-1;bool half=false;};
int assaultCells[13],assaultNeighbors[13][4],assaultCount=0,assaultEnemy=-1;
int assaultHome=-1,assaultWork=0;

void assaultMove(AssaultState &s,const AssaultOrder &order,int player)const {
 if(order.s<0||s.owner[order.s]!=player||s.owner[0]==p)return;
 int moving=order.half?s.army[order.s]/2:std::max(0,s.army[order.s]-1);
 s.army[order.s]-=moving;
 if(s.owner[order.t]==player)s.army[order.t]+=moving;
 else if(moving>s.army[order.t]){s.army[order.t]=moving-s.army[order.t];s.owner[order.t]=player;}
 else s.army[order.t]-=moving;
}

int assaultReplies(const AssaultState &s,AssaultOrder *orders)const {
 int count=1;orders[0]={-1,-1,false};
 for(int u=0;u<assaultCount;u++)if(s.owner[u]==assaultEnemy&&s.army[u]>1){
  for(int k=0;k<4;k++){
   int v=assaultNeighbors[u][k];if(v<0)continue;
   orders[count++]={u,v,false};if(s.army[u]>=4)orders[count++]={u,v,true};
  }
 }
 return count;
}

bool assaultHomeAlive(const AssaultState &s)const {return assaultHome<0||s.owner[assaultHome]==p;}

bool assaultFinish(const AssaultState &s){
 if(s.owner[0]==p)return true;
 AssaultOrder replies[105];int count=assaultEnemy<p?assaultReplies(s,replies):1;
 for(int k=0;k<4;k++){
  int source=assaultNeighbors[0][k];
  if(source<0||s.owner[source]!=p||s.army[source]-1<=s.army[0])continue;
  if(assaultEnemy>p)return true;
  bool safe=true;
  for(int j=0;j<count&&safe;j++){
   if(++assaultWork>20000)return false;
   AssaultState next=s;assaultMove(next,replies[j],assaultEnemy);
   if(!assaultHomeAlive(next)){safe=false;break;}
   assaultMove(next,{source,0,false},p);
   safe=next.owner[0]==p;
  }
  if(safe)return true;
 }
 return false;
}

void coordinatedAssaultPrograms(){
 if(aliveCount!=2)return;
 assaultWork=0;
 for(int enemy=0;enemy<o->players;enemy++){
  int goal=enemyGeneral[enemy];
  if(enemy==p||!o->alive[enemy]||goal<0||!o->visible[goal])continue;
  int attackers=0;
  for(int k=0;k<4;k++){int c=nb[goal][k];if(c>=0&&mine(c)&&ready(c)>3)attackers++;}
  if(attackers<2)continue;
  assaultCount=1;assaultCells[0]=goal;assaultEnemy=enemy;assaultHome=-1;
  for(int c=0;c<N;c++)if(c!=goal&&!blocked(c)&&
     std::abs(cellX[c]-cellX[goal])+std::abs(cellY[c]-cellY[goal])<=2)assaultCells[assaultCount++]=c;
  AssaultState initial;
  for(int j=0;j<assaultCount;j++){
   int c=assaultCells[j];initial.owner[j]=own[c];initial.army[j]=mine(c)?ready(c):rememberedDefense(c);
   if(c==g)assaultHome=j;
   for(int k=0;k<4;k++){
    assaultNeighbors[j][k]=-1;
    for(int z=0;z<assaultCount;z++)if(assaultCells[z]==nb[c][k])assaultNeighbors[j][k]=z;
   }
  }
  AssaultOrder replies[105];int count=assaultReplies(initial,replies);
  for(int k=0;k<4&&assaultWork<20000;k++){
   int source=assaultNeighbors[0][k];if(source<0||initial.owner[source]!=p||initial.army[source]<4)continue;
   int c=assaultCells[source];Order order{c,goal,false};
   if(dangerAfter(&order)>baselineDanger+1)continue;
   bool safe=true;
   for(int j=0;j<count&&safe;j++){
    if(++assaultWork>20000){safe=false;break;}
    AssaultState next=initial;
    if(enemy<p)assaultMove(next,replies[j],enemy);
    if(!assaultHomeAlive(next)){safe=false;break;}
    assaultMove(next,{source,0,false},p);
    if(enemy>p)assaultMove(next,replies[j],enemy);
    if(!assaultHomeAlive(next)){safe=false;break;}
    if(next.owner[0]==p)continue;
    for(int z=0;z<assaultCount;z++)if(next.owner[z]>=0)next.army[z]+=grows(assaultCells[z],actionTurn(2));
    if(!assaultFinish(next))safe=false;
   }
   if(safe)offer(oneStep({50000.,c,goal,direction(c,goal),false},17));
  }
 }
}

double campaignCost[MAXN];
bool campaignClosed[MAXN];
std::vector<std::pair<double,int>> campaignHeap;

void campaignSearch(int source,int initial,int limit){
 if(tick<150&&o->total_cities[p]==0&&ed[source]>=INF){search(source,initial,limit);return;}
 std::fill(dist,dist+N,INF);std::fill(force,force+N,-INF);
 std::fill(campaignCost,campaignCost+N,1e30);
 std::fill(campaignClosed,campaignClosed+N,false);
 std::fill(searchTargets,searchTargets+(N+63)/64,0);
 double income=.5*(std::max<int64_t>(0,o->total_cities[p])+1)+.02*o->total_land[p];
 double lossUnit=income*2.5+2;
 using Entry=std::pair<double,int>;
 auto &heap=campaignHeap;heap.clear();
 auto push=[&](Entry entry){heap.push_back(entry);std::push_heap(heap.begin(),heap.end(),std::greater<Entry>());};
 dist[source]=0;force[source]=initial;pathLoss[source]=0;first[source]=-1;parent[source]=-1;pathExposure[source]=0;
 campaignCost[source]=0;push({0,source});
 while(!heap.empty()){
  std::pop_heap(heap.begin(),heap.end(),std::greater<Entry>());auto [cost,u]=heap.back();heap.pop_back();
  if(campaignClosed[u]||cost>campaignCost[u]+1e-8)continue;
  campaignClosed[u]=true;if(dist[u]>=limit)continue;
  for(int k=0;k<4;k++){
   int v=nb[u][k];if(v<0||blocked(v)||campaignClosed[v])continue;
   int d=dist[u]+1,sent=force[u],locked=0;bool half=false;
   if(u!=source){
    int available=force[u]+grows(u,actionTurn(d));half=reserve[u]>1;
    sent=half?available/2:available-1;
    if(available-sent<reserve[u])continue;
    locked=std::max(0,available-sent-(mine(u)?army(u):1));
   }
   int guard=mine(v)?0:rememberedDefense(v);
   int period=(st[v]>0&&st[v]<3)?2:50;
   int growth=std::max(0,actionTurn(d)/period-actionTurn(1)/period);
   if(own[v]>=0&&!mine(v))guard+=growth;
   int f=sent+(mine(v)?ready(v)+growth:-guard-int(ter[v]==255));if(f<1)continue;
   // Fighting another army removes troops from both sides; buying a neutral
   // city removes only ours. Keep actual force propagation unchanged and
   // distinguish these opportunity costs only when ranking approach routes.
   double price=1.;
   if(own[v]>=0&&own[v]!=p)
    price=std::clamp(.5+double(o->total_army[own[v]]-o->total_army[p])/(o->total_army[p]+30.),.15,1.1);
   if(aliveCount>2)price=1.-(1.-price)/(aliveCount-1);
   int spent=pathLoss[u]+int(u!=source)+int(std::lround(guard*price))+
     int(ter[v]==255)+int(locked*.3);
   double exposure=std::max(pathExposure[u],landingExposure(u,v,f,parent,dist));
   double next=dist[u]+1+spent/lossUnit+exposure;
   if(next>campaignCost[v]+1e-8||(std::abs(next-campaignCost[v])<=1e-8&&f<=force[v]))continue;
   campaignCost[v]=next;dist[v]=dist[u]+1;force[v]=f;pathLoss[v]=spent;
   first[v]=u==source?k:first[u];parent[v]=u;pathExposure[v]=exposure;
   pathHalf[v]=half;
   if(!mine(v))searchTargets[v>>6]|=std::uint64_t(1)<<(v&63);
   push({next,v});
  }
 }
}

// Spend a limited number of orders revealing likely hostile back country
// with expendable edge troops. Information is priced as a fraction of the
// remaining general-location distribution, rather than per fog cell forever.
void frontierScoutPrograms(){
 if(tick<200||expeditionCredit<1||baselineDanger>0)return;
 bool unknownKing=false;double prize=0,total=0;
 for(int e=0;e<o->players;e++)if(e!=p&&o->alive[e]&&enemyGeneral[e]<0){
  unknownKing=true;prize=std::max(prize,eliminationValue(e));
 }
 if(!unknownKing)return;
 for(int c=0;c<N;c++)total+=intel[c];
 if(total<1)return;
 for(int source=0;source<N;source++)if(mine(source)&&source!=g&&ready(source)>=4){
  bool half=false;if(!canDepart(source,half))continue;
  if(ready(source)>28||st[source]==1)half=true;
  int moving=moveArmy(source,half);if(moving<3||moving>40)continue;
  for(int k=0;k<4;k++){
   int target=nb[source][k];if(target<0||blocked(target)||mine(target)||intelGain[target]<1.)continue;
   int guard=rememberedDefense(target),left=moving-guard;
   if(guard>3||left<2)continue;
   if(zoneETA[target]<=2&&zoneThreat[target]>left+2)continue;
   Order x{source,target,half};if(!permitted(x))continue;
   double information=std::min(10.,prize*.55*intelGain[target]/total);
   double value=(2.+(own[target]>=0?1.5:0)+information)/(1.5+.025*moving);
   value-=risk(source,target,moving);
   offer(oneStep({value,source,target,k,half},18));
  }
 }
}

// A garrison just below twice its reserve cannot legally dispatch half.
// Reinforcing it with a small, disjoint donor tree can unlock a much larger
// mobile column. Price only that additional deployable force, and preserve
// the general's proactive defense target so readiness will not recall it.
void unlockGarrisonPrograms(){
 if(tick<180||baselineDanger>0)return;
 for(int root=0;root<N;root++)if(mine(root)&&st[root]>0&&st[root]<3&&ready(root)>=40){
  int guard=reserve[root];
  if(root==g&&ed[g]<=8&&homeIncoming[8]>=8)
   guard=std::max(guard,std::min(int(o->total_army[p]*.45),int(homeIncoming[8]*1.35)+6));
  if(guard<=1||(ready(root)+1)/2>=guard||ready(root)<guard)continue;
  scheduleTree(root,8);Program best;
  for(int k=1;k<=8;k++){
   int assembled=schedule[root][k];if((assembled+1)/2<guard+2)continue;
   Program q;q.kind=19;q.root=q.goal=root;q.born=tick;
   extractSchedule(root,k,q);
   int deployable=assembled/2;
   int useful=mobileAssemblyCredit(q,root,deployable+1,deployable+1);
   if(useful<8)continue;
   q.value=std::min(8.,.9*useful/(k+1.5+.20*std::min(20,ed[root])));
   if(q.value>best.value&&q.count&&permitted(q.orders[0]))best=q;
  }
  offer(best);
 }
}

struct SiegeRoute {int source=-1,count=0,arrival=0;bool half=false;int path[10];};
void convergingAssaultPrograms(){
 if(aliveCount!=2)return;
 for(int enemy=0;enemy<o->players;enemy++){
  int goal=enemyGeneral[enemy];if(enemy==p||!o->alive[enemy]||goal<0||!o->visible[goal])continue;
  SiegeRoute routes[8];int count=0;
  for(int source:sources){
   bool half=false;if(!canDepart(source,half)||moveArmy(source,half)<15)continue;
   if(count==8)break;
   std::fill(dist,dist+N,INF);std::fill(force,force+N,-INF);
   int head=0,tail=1;queue[0]=source;dist[source]=0;force[source]=moveArmy(source,half);parent[source]=-1;
   while(head<tail){
    int u=queue[head++];if(u==goal||dist[u]>=9)continue;
    for(int k=0;k<4;k++){
     int v=nb[u][k];if(v<0||v==source||blocked(v)||dist[v]<dist[u]+1)continue;
     if(mine(v)&&reserve[v]>1)continue;
     int arriving=force[u]-int(u!=source)+(v==goal?0:routeDelta[v]);
     if(arriving<1)continue;
     if(dist[v]==INF){dist[v]=dist[u]+1;queue[tail++]=v;}
     if(arriving>force[v]){force[v]=arriving;parent[v]=u;}
    }
   }
   if(dist[goal]>=INF||force[goal]<10)continue;
   SiegeRoute route;route.source=source;route.half=half;route.arrival=force[goal];
   int reverse[10],n=0;for(int c=goal;c!=source&&c>=0&&n<10;c=parent[c])reverse[n++]=c;
   for(int j=n-1;j>=0;j--)route.path[route.count++]=reverse[j];
   routes[count++]=route;
  }
  for(int a=0;a<count;a++)for(int b=0;b<count;b++)if(a!=b){
   const auto &first=routes[a],&second=routes[b];int orders=first.count+second.count;
   if(orders>16)continue;
   bool disjoint=first.source!=second.source;
   for(int i=-1;i<first.count-1;i++)for(int j=-1;j<second.count-1;j++)
    if((i<0?first.source:first.path[i])==(j<0?second.source:second.path[j]))disjoint=false;
   if(!disjoint)continue;
   int defense=rememberedDefense(goal)+kingRelief[enemy][orders]+futureGrowth(orders)+8;
   if(first.arrival+second.arrival<=defense)continue;
   int next=first.path[0];Order action{first.source,next,first.half};
   if(next!=goal&&!permitted(action))continue;
   if(next==goal&&dangerAfter(&action)>baselineDanger+1)continue;
   double value=eliminationValue(enemy)/(2.+.5*orders);
   offer(oneStep({value,action.s,action.t,direction(action.s,action.t),action.half},21));
  }
 }
}

int collectionTroops[MAXN];
int collectionArrival(const Program &q,int root){
 int cells[41],count=1;cells[0]=root;
 for(int j=0;j<q.count;j++){
  int c=q.orders[j].s;bool found=false;for(int k=0;k<count;k++)if(cells[k]==c)found=true;
  if(!found)cells[count++]=c;
 }
 for(int j=0;j<count;j++)collectionTroops[cells[j]]=ready(cells[j]);
 for(int j=0;j<q.count;j++){
  if(j)for(int k=0;k<count;k++)collectionTroops[cells[k]]+=grows(cells[k],actionTurn(j+1));
  auto x=q.orders[j];int sent=x.half?collectionTroops[x.s]/2:collectionTroops[x.s]-1;
  if(sent<1||collectionTroops[x.s]-sent<reserve[x.s])return -INF;
  collectionTroops[x.s]-=sent;collectionTroops[x.t]+=sent;
 }
 return collectionTroops[root]+grows(root,actionTurn(q.count+1));
}

// Productive cities turn income into a column without harvesting dozens of
// separate land tiles. Purchase plans start at the actual entrance, include
// every collection order, and forecast growth during that concrete schedule.
void productionInvestmentPrograms(){
 if(tick<85||baselineDanger>0)return;
 int cities[6],count=0;double priorities[6];
 for(int c=0;c<N;c++)if(st[c]==1&&own[c]<0&&!blocked(c)&&zoneETA[c]>5){
  int best=0;for(int k=0;k<4;k++){int v=nb[c][k];if(v>=0&&mine(v))best=std::max(best,ready(v));}
  if(!best)continue;
  double value=best/(rememberedDefense(c)+1.)+1./(gd[c]+1.);
  int j=count;if(j==6){j=5;if(value<=priorities[j])continue;}else count++;
  while(j>0&&value>priorities[j-1]){if(j<6){cities[j]=cities[j-1];priorities[j]=priorities[j-1];}j--;}
  cities[j]=c;priorities[j]=value;
 }
 double income=.5*(std::max<int64_t>(0,o->total_cities[p])+1)+.02*o->total_land[p];
 for(int ci=0;ci<count;ci++){
  int city=cities[ci],root=-1,best=0;
  for(int k=0;k<4;k++){
   int c=nb[city][k];if(c<0||!mine(c))continue;
   int available=ready(c);for(int j=0;j<4;j++){int v=nb[c][j];if(v>=0&&mine(v))available+=std::max(0,ready(v)-1);}
   if(available>best){best=available;root=c;}
  }
  if(root<0)continue;
  scheduleTree(root,20);Program bestPlan;
  for(int k=1;k<=20;k++)if(schedule[root][k]>0){
   Program q;q.root=root;q.goal=city;q.kind=22;q.born=tick;extractSchedule(root,k,q);
   if(!q.count)continue;
   int assembled=collectionArrival(q,root);if(assembled<1)continue;
   bool half=reserve[root]>1;int mobile=half?assembled/2:assembled-1;
   if(assembled-mobile<reserve[root])continue;
   int remaining=mobile-rememberedDefense(city);if(remaining<6)continue;
   if(moveArmy(root,half)>rememberedDefense(city)+5)continue;
   double continuation=capitalHorizon/(2.+std::max<int64_t>(0,o->total_cities[p]));
   q.value=(cityUtility(city,remaining,q.count+1)+continuation)/
     (q.count+1.8+rememberedDefense(city)/(income*2.5+2));
   q.orders[q.count++]={root,city,half};
   if(q.value>bestPlan.value&&permitted(q.orders[0]))bestPlan=q;
  }
  offer(bestPlan);
 }
}
