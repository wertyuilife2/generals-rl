// Strategic state belongs to this seat. Tasks survive score changes and are
// cancelled only by completion, infeasibility, deadlines, or tactical danger.
enum TaskKind { NO_TASK, CONTROL, RECON, STRIKE };
struct Task {
 int kind=NO_TASK, goal=-1, column=-1, born=0, deadline=0, need=0;
 int gathering=0, lastProgress=0, startDistance=INF;
 Program delivery;
};
Task operation,scoutTask;
int taskOwner[MAXN]{},assetHeldSince[MAXN]{},assetLostAt[MAXN]{},assetFlips[MAXN]{};
int assetHeldTotal[MAXN]{},assetCaptures[MAXN]{},assetLastOwner[MAXN]{};
bool assetHistoryReady=false;
int assetGuard[MAXN]{},assetETA[MAXN]{},zoneThreat[MAXN]{},zoneETA[MAXN]{};
double intel[MAXN]{},intelGain[MAXN]{},routeCost[MAXN];
int opParent[MAXN],opSteps[MAXN],opForce[MAXN];
int pendingTask=0,pendingSource=-1,pendingDest=-1,pendingSent=0,pendingBefore=0,pendingAt=-1;
int lastScout=-1000,lastOperation=-1000,lastTaskPlan=-1000,operationFailures=0;
std::int64_t pendingAction=0;

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

// A bounded reach envelope describes observed and recently hidden armies.
// It does not invent the location of an unobserved enemy army.
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
 // Cities receive a real retention requirement when affordable. A city that
 // cannot be held is evacuated, not repeatedly garrisoned below the deadline.
 for(int c=0;c<N;c++){
  assetGuard[c]=1;assetETA[c]=zoneETA[c];
  if(st[c]!=1||!mine(c)||zoneETA[c]>5)continue;
  int need=zoneThreat[c]-futureGrowth(std::max(0,zoneETA[c]))+2;
  int support=0;
  for(int k=0;k<4;k++){int v=nb[c][k];if(v>=0&&mine(v)&&v!=g)support+=std::max(0,army(v)-1);}
  bool viable=need<=army(c)+support&&need<=std::max(20,int(o->total_army[p]*.4));
  if(viable){assetGuard[c]=std::max(1,need);reserve[c]=assetGuard[c];}
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
 return value;
}

void intelligenceMap(){
 std::fill(intel,intel+N,0.);std::fill(intelGain,intelGain+N,0.);
 bool kingUnknown=false;int enemyTiles=0;
 double ex=0,ey=0;
 for(int e=0;e<o->players;e++)if(e!=p&&o->alive[e]&&enemyGeneral[e]<0)kingUnknown=true;
 for(int c=0;c<N;c++)if(own[c]>=0&&own[c]!=p&&o->alive[own[c]]){
  double w=st[c]==1?3.:1.;ex+=cellX[c]*w;ey+=cellY[c]*w;enemyTiles+=int(w);
 }
 if(enemyTiles){ex/=enemyTiles;ey/=enemyTiles;}
 else {ex=W-1-cellX[g];ey=H-1-cellY[g];}
 for(int c=0;c<N;c++){
  if(ter[c]==1)continue;
  double d=std::abs(cellX[c]-ex)+std::abs(cellY[c]-ey);
  // A general cannot move. Previously observed non-general cells are excluded
  // from its spatial prior, while stale production ownership still has value.
  if(kingUnknown&&seen[c]<0&&ter[c]!=2){
   double away=std::clamp((gd[c]-6.)/12.,.10,1.);
   intel[c]=away*(enemyTiles?1.8/(1+.045*d):1.0);
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

// Risk-aware paths can deliberately take a longer route around a main army.
// Work is bounded by board size; friendly collection is accounted once.
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

bool permitted(const Order &x,bool emergency=false){
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

// Every donor path strictly approaches one fixed rally point. Troops already
// in the column are never moved out and then rewarded for merging back in.
Program collectFor(int root,int needed,int maxOrders){
 Program best;
 bfs(root,opSteps,true);
 int candidates[16],count=0;
 for(int c=0;c<N;c++)if(c!=root&&mine(c)&&opSteps[c]<=maxOrders&&opSteps[c]>0){
  bool half=false;if(!canDepart(c,half)||moveArmy(c,half)<3)continue;
  double score=double(moveArmy(c,half))/(opSteps[c]+1);int j=count;
  if(j==16){j=15;bool hh=false;canDepart(candidates[j],hh);if(score<=double(moveArmy(candidates[j],hh))/(opSteps[candidates[j]]+1))continue;}else count++;
  while(j>0){bool hh=false;canDepart(candidates[j-1],hh);if(score<=double(moveArmy(candidates[j-1],hh))/(opSteps[candidates[j-1]]+1))break;candidates[j]=candidates[j-1];j--;}
  candidates[j]=c;
 }
 for(int j=0;j<count;j++){
  int s=candidates[j],u=s,force=ready(s),cost=0;Program q;bool failed=false;
  q.root=root;q.goal=root;q.kind=2;q.born=tick;
  while(u!=root&&cost<maxOrders){
   int next=-1;
   for(int k=0;k<4;k++){int v=nb[u][k];if(v>=0&&mine(v)&&opSteps[v]+1==opSteps[u]&&(next<0||army(v)>army(next)))next=v;}
   if(next<0){failed=true;break;}
   bool half=reserve[u]>1;
   if(half&&(force+1)/2<reserve[u]){failed=true;break;}
   int sent=half?force/2:force-1;
   if(sent<=0){failed=true;break;}
   q.orders[q.count++]={u,next,half};cost++;
   force=sent+army(next);u=next;
  }
  if(failed||u!=root||q.count==0||!permitted(q.orders[0]))continue;
  int gained=force-ready(root);
  q.value=double(std::min(std::max(0,needed),gained))/(cost+1.0);
  if(q.value>best.value)best=q;
 }
 return best;
}

void acknowledgeTasks(){
 if(pendingTask<=0)return;
 bool receipt=o->previous&&o->previous->action==pendingAction&&o->previous->valid;
 bool inferred=!o->previous&&mine(pendingDest)&&
  (!mine(pendingSource)||army(pendingSource)<pendingBefore-pendingSent/2);
 if(receipt||inferred){
  Task &t=pendingTask==1?operation:scoutTask;
  if(t.column==pendingSource)t.column=pendingDest;
  if(t.delivery.cursor<t.delivery.count){
   const auto &x=t.delivery.orders[t.delivery.cursor];
   if(x.s==pendingSource&&x.t==pendingDest)t.delivery.cursor++;
  }
  t.lastProgress=tick;
  pendingTask=0;
 }else if(tick-pendingAt>2||(o->previous&&o->previous->action==pendingAction)){
  Task &t=pendingTask==1?operation:scoutTask;t.delivery=Program();pendingTask=0;
 }
}

bool taskAlive(Task &t,bool scout=false){
 if(t.kind==NO_TASK)return false;
 if(t.goal<0||t.column<0||tick>t.deadline||blocked(t.goal)||!mine(t.column))return false;
 if(t.kind==CONTROL&&mine(t.goal))return false;
 if(t.kind==STRIKE&&(own[t.goal]<0||!o->alive[own[t.goal]]))return false;
 if(t.kind==RECON&&(intelGain[t.goal]<1.5||t.column==t.goal))return false;
 if(tick-t.lastProgress>(scout?24:18))return false;
 return true;
}

int mobileDemand(int goal,int distance)const {
 int enemy=own[goal],visible=zoneThreat[goal];
 int publicMobile=0;
 for(int e=0;e<o->players;e++)if(e!=p&&o->alive[e])publicMobile=std::max(publicMobile,int(std::max<int64_t>(0,o->total_army[e]-o->total_land[e])));
 int need=std::max(20,rememberedDefense(goal)+distance+8);
 if(st[goal]==1)need=std::max(need,visible+distance+8);
 else if(st[goal]==2&&enemy>=0)need+=std::min(publicMobile/3,100)+distance/2;
 else need=std::max(need,std::min(visible+12,publicMobile/3+20));
 int ownMobile=std::max<int64_t>(0,o->total_army[p]-o->total_land[p]);
 return std::min(need,std::max(24,int(ownMobile*.65)));
}

Task chooseTask(bool scout){
 Task best;double bestValue=-1e20;
 int roots[8],n=0;
 // Large armies and useful front positions are distinct candidate sources.
 for(int c:sources){
  bool half=false;if(!canDepart(c,half))continue;
  int mobile=moveArmy(c,half);
  if(scout&&(c==operation.column||mobile<5||mobile>std::max(40,int(o->total_army[p]*.18))))continue;
  roots[n++]=c;if(n==(scout?5:6))break;
 }
 for(int ri=0;ri<n;ri++){
  int source=roots[ri];bool half=false;if(!canDepart(source,half))continue;
  int initial=moveArmy(source,half);
  taskRoutes(source,scout?initial:std::max(initial,int((o->total_army[p]-o->total_land[p])*.65)),scout);
  for(int c=0;c<N;c++){
   if(c==source||opSteps[c]>=INF||opSteps[c]>32||blocked(c))continue;
   int kind=NO_TASK;double reward=0;
   bool enemy=own[c]>=0&&own[c]!=p&&o->alive[own[c]];
   if(!scout&&st[c]==2&&enemy){kind=STRIKE;reward=80.+std::sqrt(double(o->total_army[own[c]]))*.6;}
   else if(!scout&&st[c]==1&&!mine(c)){
    kind=CONTROL;reward=assetUtility(c,opForce[c],opSteps[c])*2.;
    if(reward<=3)continue;
   }else if(intelGain[c]>=2.0&&!o->visible[c]&&st[c]!=1){
    kind=RECON;reward=intelGain[c]*(scout?4.0:2.3);
    if(enemy)reward+=2.;
   }else continue;
   if(scout&&kind!=RECON)continue;
   int needed=scout?initial:mobileDemand(c,opSteps[c]);
   double assembly=std::max(0,needed-initial)/std::max(3.,double(o->total_army[p]-o->total_land[p])/20.);
   double value=reward/(4.+routeCost[c]+assembly);
   if(!scout&&kind==RECON&&army(source)<15)value*=.6;
   if(value<=bestValue)continue;
   bestValue=value;best.kind=kind;best.goal=c;best.column=source;best.born=tick;
   best.deadline=tick+std::min(100,36+opSteps[c]*2);best.need=needed;best.gathering=scout?0:1;
   best.lastProgress=tick;best.startDistance=opSteps[c];
  }
 }
 return best;
}

Order executeTask(Task &t,bool scout){
 Order none;
 if(!taskAlive(t,scout)){t=Task();return none;}
 if(t.delivery.cursor<t.delivery.count){
  Order x=t.delivery.orders[t.delivery.cursor];
  if(permitted(x))return x;
  t.delivery=Program();
 }
 int s=t.column;bool half=false;if(!canDepart(s,half))return none;
 int initial=moveArmy(s,half);
 // Assembly consumes at most fourteen actual actions before advancing; it
 // cannot consume an entire game waiting for an unreachable desired army.
 if(t.gathering&&initial<t.need&&tick-t.born<18){
  t.delivery=collectFor(s,t.need-initial,10);
  if(t.delivery.count){t.gathering++;return t.delivery.orders[0];}
 }
 t.gathering=0;
 taskRoutes(s,initial,scout);
 if(opSteps[t.goal]>=INF){
  if(!scout&&tick-t.lastProgress<=8){t.delivery=collectFor(s,std::max(20,t.need-initial),8);if(t.delivery.count)return t.delivery.orders[0];}
  t=Task();return none;
 }
 int next=routeFirst(s,t.goal);if(next<0){t=Task();return none;}
 Order x{s,next,half};
 if(!permitted(x))return none;
 // A scout's life is priced against information, not used to hit a visible
 // main army whose counterattack reaches the destination immediately.
 int residual=initial+(mine(next)?army(next):-rememberedDefense(next));
 if(scout&&zoneETA[next]<=2&&zoneThreat[next]>residual+5)return none;
 return x;
}

Order protectAssets(){
 Order best;double score=0;
 for(int c=0;c<N;c++)if(mine(c)&&st[c]==1&&assetGuard[c]>army(c)+1&&assetETA[c]<=5){
  int lack=assetGuard[c]-army(c),budget=std::max(1,assetETA[c]-1);
  Program q=collectFor(c,lack,budget);
  if(q.count==0)continue;
  double value=(std::min(60,lack)+15.)/(q.count+1.0)/(assetETA[c]+1.);
  if(value>score){score=value;best=q.orders[0];}
 }
 // Capturing an exposed raider protects production and frees future actions.
 for(int s=0;s<N;s++)if(mine(s)&&ready(s)>3)for(int k=0;k<enemyCount[s];k++){
  int t=enemyNeighbors[s][k];if(army(t)<5)continue;
  bool relevant=false;
  for(int j=0;j<4;j++){int v=nb[t][j];if(v>=0&&mine(v)&&(st[v]==1||v==g))relevant=true;}
  if(!relevant)continue;
  for(int half=0;half<2;half++){
   int sent=moveArmy(s,half);Order x{s,t,bool(half)};
   if(sent<=rememberedDefense(t)+2||!permitted(x))continue;
   int residual=sent-rememberedDefense(t);
   if(zoneThreat[t]>residual+sent&&zoneETA[t]<=1)continue;
   double value=4.+std::min(80,army(t))*.16;
   if(value>score){score=value;best=x;}
  }
 }
 return best;
}

std::int64_t dispatch(const Order &x,int task=0){
 if(x.s<0||x.t<0)return 0;
 auto action=arena_move(x.s,direction(x.s,x.t),x.half);
 if(task){
  pendingTask=task;pendingSource=x.s;pendingDest=x.t;pendingSent=moveArmy(x.s,x.half);
  pendingBefore=army(x.s);pendingAt=tick;pendingAction=action;
  if(task==2)lastScout=tick;else lastOperation=tick;
 }
 recordFlow(x,action);oldSource=x.t;lastTarget=x.t;
 return action;
}

std::int64_t decideV8(const ArenaObservation *obs){
 if(obs->width*obs->height>MAXN||obs->general_pos<0)return 0;
 update(obs);assetHistory();pressureMap();supplyMap();intelligenceMap();acknowledgeTasks();
 buildKingRelief();buildAttacks();baselineDanger=dangerAfter(nullptr,homeIncoming);proposalCount=0;
 for(int s=0;s<N;s++)if(mine(s))for(int k=0;k<enemyCount[s];k++){
  int t=enemyNeighbors[s][k];if(st[t]!=2)continue;
  for(int half=0;half<2;half++)if(moveArmy(s,half)>captureDefense(t)&&(aliveCount<=2||captureSafe(s,t,moveArmy(s,half))))return dispatch({s,t,bool(half)});
 }
 Candidate defense=defend(),forecast=forecastDefend();if(forecast.score>defense.score)defense=forecast;
 if(defense.source>=0&&defense.score>15){
  Order x{defense.source,nb[defense.source][defense.dir],defense.half};if(legalOrder(x,true))return dispatch(x);
 }
 homeCounterPrograms();
 for(int j=0;j<proposalCount;j++)if(proposals[j].kind==9)return dispatch(proposals[j].orders[0]);
 Order protect=protectAssets();if(protect.s>=0)return dispatch(protect);
 sources.clear();
 for(int c=0;c<N;c++)if(mine(c)&&canMoveNow(c))sources.push_back(c);
 std::sort(sources.begin(),sources.end(),[&](int x,int y){return army(x)>army(y);});
 // Retain frontier donors and cheap reconnaissance armies alongside big stacks.
 keep.clear();for(int c:sources)if(keep.size()<8)keep.push_back(c);
 border.clear();for(int c:sources)if(std::find(keep.begin(),keep.end(),c)==keep.end())border.emplace_back(ed[c],c);
 std::sort(border.begin(),border.end());for(int j=0;j<std::min(8,int(border.size()));j++)keep.push_back(border[j].second);
 sources=keep;
 if(!taskAlive(operation)){operation=Task();}
 if(!taskAlive(scoutTask,true)){scoutTask=Task();}
 bool strategic=tick>=200&&o->total_land[p]>=35;
 if(strategic&&operation.kind==NO_TASK&&tick-lastTaskPlan>=3){operation=chooseTask(false);lastTaskPlan=tick;}
 if(strategic&&scoutTask.kind==NO_TASK&&tick-lastScout>=6)scoutTask=chooseTask(true);
 if(scoutTask.kind!=NO_TASK&&tick-lastScout>=6){Order x=executeTask(scoutTask,true);if(x.s>=0)return dispatch(x,2);}
 if(operation.kind!=NO_TASK&&tick%3!=0){Order x=executeTask(operation,false);if(x.s>=0)return dispatch(x,1);}
 // Expansion remains a bounded path problem. Strategic tasks own their action
 // slots; a nearby city reward cannot replace an accepted collection program.
 proposalCount=0;
 double income=.5*(std::max<int64_t>(0,o->total_cities[p])+1)+.02*o->total_land[p];
 campaignPrograms(income);routePlan(income);
 if(tick>=48)logisticsPrograms(income);
 readinessProgram();deploymentPrograms();interceptionPrograms();
 int best=-1;double value=-.01-baselineDanger;
 for(int j=0;j<proposalCount;j++)if(proposalScore[j]>value&&permitted(proposals[j].orders[0])){best=j;value=proposalScore[j];}
 if(best>=0){auto &q=proposals[best];oldTarget=q.goal;return dispatch(q.orders[q.cursor]);}
 return 0;
}
