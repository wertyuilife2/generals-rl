// Strategic estimates use only this seat's observation history. They feed
// executable programs into the common action scheduler in controller.h.
int assetHeldSince[MAXN]{},assetLostAt[MAXN]{},assetFlips[MAXN]{};
int assetHeldTotal[MAXN]{},assetCaptures[MAXN]{},assetLastOwner[MAXN]{};
bool assetHistoryReady=false;
int assetGuard[MAXN]{},assetETA[MAXN]{},zoneThreat[MAXN]{},zoneETA[MAXN]{};
double intel[MAXN]{},intelGain[MAXN]{},routeCost[MAXN];
int opParent[MAXN],opSteps[MAXN],opForce[MAXN];

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
 // Retention is a portfolio decision: several individually defensible
 // cities must not each claim the same field army or consume the entire army.
 int candidates[MAXN],countCities=0;
 double priority[MAXN];
 int mobile=std::max<int64_t>(0,o->total_army[p]-o->total_land[p]);
 int guardBudget=std::max(12,int(mobile*.25));
 for(int c=0;c<N;c++){
  assetGuard[c]=1;assetETA[c]=zoneETA[c];
  if(st[c]!=1||!mine(c)||zoneETA[c]>5)continue;
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
 int local=0;
 for(int k=0;k<enemyCount[c];k++)local=std::max(local,moveArmy(enemyNeighbors[c][k]));
 double holding=std::min(80.,16.+8.*ed[c]);
 if(local>remaining+3)holding=std::min(holding,8.);
 else if(remaining>local+12)holding=std::min(90.,holding+16.);
 int pressure=zoneThreat[c],eta=zoneETA[c];
 if(pressure>remaining+3&&eta<=6)holding=std::min(holding,double(std::max(2,eta-arrival+2)));
 if(assetFlips[c]>0&&remaining<pressure+20)holding=std::min(holding,24./(1+.4*assetFlips[c]));
 int exits=0;for(int k=0;k<4;k++){int v=nb[c][k];if(v>=0&&!blocked(v)&&st[v]!=1)exits++;}
 double value=(5.+holding*.29+exits*.55)/(1+.012*arrival);
 if(own[c]>=0&&own[c]!=p)value+=6.*std::min(1.,holding/24.);
 if(remaining<4)value*=.70;
 if(assetFlips[c]>0&&remaining<pressure+20)value-=std::min(10.,1.5*assetFlips[c]);
 return value;
}

void intelligenceMap(){
 std::fill(intel,intel+N,0.);std::fill(intelGain,intelGain+N,0.);
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
  if(!o->visible[c]&&st[c]==1)intel[c]+=.3*std::min(1.,(tick-seen[c])/40.);
  if(!o->visible[c]&&ter[c]==2)intel[c]+=.18;
 }
 for(int c=0;c<N;c++)if(!blocked(c)){
  for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){
   int x=cellX[c]+dx,y=cellY[c]+dy;if(x>=0&&x<W&&y>=0&&y<H)intelGain[c]+=intel[y*W+x];
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
   int guard=mine(v)?0:rememberedDefense(v),available=opForce[u]-1;
   if(mine(v))available+=std::max(0,army(v)-reserve[v]);else available-=guard;
   if(available<2&&!friendly)continue;
   int mobileThreat=zoneThreat[v];
   double riskCost=std::max(0,mobileThreat-available)* (recon?.14:.04);
   if(recon&&zoneETA[v]<=2&&mobileThreat>available+5)riskCost+=8.;
   double next=cost+1.+guard/(recon?5.:18.)+riskCost+(ter[v]==255?.20:0.);
   if(next+1e-6>=routeCost[v])continue;
   routeCost[v]=next;opParent[v]=u;opSteps[v]=opSteps[u]+1;opForce[v]=std::max(1,available);heap.push({next,v});
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
 if(tick<260||baselineDanger>0||o->total_land[p]<40||o->total_cities[p]<2)return;
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
  int u=source;for(int j=n-1;j>=0;j--){q.orders[q.count++]={u,path[j],j==n-1&&half};u=path[j];}
  offer(q);
 }
}

void mobilizationPrograms(){
 if(tick<260||baselineDanger>0||o->total_cities[p]<2)return;
 int roots=0;
 for(int root:sources){
  if(root==g||ed[root]>10||army(root)<4)continue;
  if(zoneETA[root]<=4&&zoneThreat[root]>=ready(root)-2)continue;
  if(++roots>4)break;
  int need=24;
  for(int e=0;e<o->players;e++)if(e!=p&&o->alive[e])
   need=std::max(need,int((o->total_army[e]-o->total_land[e])*.35));
  for(int c=0;c<N;c++)if(o->visible[c]&&own[c]>=0&&own[c]!=p&&
    std::abs(cellX[c]-cellX[root])+std::abs(cellY[c]-cellY[root])<=10)need=std::max(need,army(c)+12);
  need=std::min(need,int((o->total_army[p]-o->total_land[p])*.65));
  if(ready(root)>=need)continue;
  int budget=tick<500?12:24;
  scheduleTree(root,budget);Program best;
  for(int k=1;k<=budget;k++)if(schedule[root][k]>ready(root)+4){
   Program q;q.root=root;q.goal=root;q.kind=11;q.born=tick;extractSchedule(root,k,q);
   int useful=assemblyCredit(q,root,schedule[root][k],need);
   q.value=.70*useful/(k+2.)/(1.+.08*ed[root]);
   q.value=9.*(-std::expm1(-q.value/9.));
   if(q.value>best.value&&q.count&&permitted(q.orders[0]))best=q;
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
