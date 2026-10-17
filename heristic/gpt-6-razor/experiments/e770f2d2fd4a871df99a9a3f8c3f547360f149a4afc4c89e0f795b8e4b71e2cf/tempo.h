// Public duel score differences cancel equal combat losses. Only consecutive
// ordinary growth boundaries are sampled; land pulses and ownership changes
// are not mistaken for recurring city production. This is an estimate, never
// access to an opponent's hidden city count or location.
int economyTick=-1,economyCities=-1,economyCursor=0,economyCount=0;
int64_t economyGap=0;
int economyAt[8]{},economyCityGap[8]{},economyLandGap[8]{};
bool economyReliable=false;
double economyMargin=0;

void observeEconomy(){
 economyReliable=false;economyMargin=0;
 if(o->players!=2||aliveCount!=2){economyCount=0;economyTick=-1;return;}
 int enemy=1-p,cities=int(std::max<int64_t>(0,o->total_cities[p]));
 int64_t gap=o->total_army[p]-o->total_army[enemy];
 bool sameCities=economyCities==cities;
 if(!sameCities){economyCount=0;economyCursor=0;}
 if(sameCities&&tick==economyTick+1&&tick%2==0&&tick%50!=0){
  int64_t change=gap-economyGap;
  // Without investment, the city-count difference cannot exceed our own
  // number of cities. Discard such positive purchase spikes. A negative
  // own purchase can only make the estimate more cautious about waiting.
  if(change<=cities&&change>=-std::max<int64_t>(1,o->total_land[enemy])){
   int i=economyCursor++%8;economyAt[i]=tick;economyCityGap[i]=int(change);
   economyLandGap[i]=int(o->total_land[p]-o->total_land[enemy]);
   economyCount=std::min(8,economyCount+1);
  }
 }
 economyTick=tick;economyGap=gap;economyCities=cities;
 int minimum=INF,count=0,hits=0,land=int(o->total_land[p]-o->total_land[enemy]);
 for(int i=0;i<economyCount;i++)if(tick-economyAt[i]<=24){
  count++;minimum=std::min(minimum,economyCityGap[i]);land=std::min(land,economyLandGap[i]);
 }
 for(int i=0;i<economyCount;i++)if(tick-economyAt[i]<=24&&economyCityGap[i]==minimum)hits++;
 // Requiring a repeated lower observation avoids interpreting a lone enemy
 // purchase as production. No unseen city is assigned to a map square.
 if(count>=4&&hits>=3){economyReliable=true;economyMargin=.5*minimum+.02*land;}
}

bool tempoActive=false;
int tempoEnemy=-1,tempoPin=-1,tempoUnseen=0,tempoTarget=0,tempoBudget=0;
Program tempoFallback;
int tempoParent[MAXN],tempoNodes[256],tempoDepth[MAXN],tempoMass[MAXN];
int tempoTreeCount=0;

// Construct one bounded owned tree after a prospective first order. All
// donors and shared edges are counted once. Future donor growth is credited
// at the response horizon, so this is an optimistic availability estimate,
// not a proof that a full adversarial rescue will execute uninterrupted.
int tempoCollection(const Order *firstOrder,int budget,Program *plan){
 if(budget<0)return -INF;
 int sent=firstOrder&&firstOrder->s>=0?moveArmy(firstOrder->s,firstOrder->half):0;
 auto ours=[&](int c){return mine(c)||(firstOrder&&c==firstOrder->t&&sent>rememberedDefense(c));};
 auto troops=[&](int c){
  int n=mine(c)?ready(c):rememberedDefense(c);
  if(firstOrder&&firstOrder->s>=0){
   if(c==firstOrder->s)n-=sent;
   if(c==firstOrder->t)n=mine(c)?n+sent:sent-n;
  }
  int period=st[c]>0&&st[c]<3?2:50;
  return n+std::max(0,actionTurn(budget+1)/period-actionTurn(1)/period);
 };
 if(!ours(g))return -INF;
 std::fill(tempoParent,tempoParent+N,-2);
 tempoTreeCount=1;tempoNodes[0]=g;tempoParent[g]=-1;tempoDepth[g]=0;
 for(int j=0;j<tempoTreeCount;j++){
  int u=tempoNodes[j];tempoMass[u]=troops(u);
  if(tempoDepth[u]>=budget)continue;
  for(int k=0;k<4;k++){
   int v=nb[u][k];if(v<0||!ours(v)||tempoParent[v]!=-2||tempoTreeCount==256)continue;
   tempoParent[v]=u;tempoDepth[v]=tempoDepth[u]+1;tempoNodes[tempoTreeCount++]=v;
  }
 }
 for(int j=tempoTreeCount-1;j>=0;j--){
  int u=tempoNodes[j];schedule[u][0]=tempoMass[u];
  for(int b=1;b<=budget;b++)schedule[u][b]=-INF;
  for(int k=0;k<4;k++){
   for(int b=0;b<=budget;b++)schedulePick[u][k][b]=-1;
   int v=nb[u][k];if(v<0||tempoParent[v]!=u)continue;
   for(int b=budget;b>=1;b--)for(int use=0;use<b;use++){
    int left=schedule[u][b-use-1],donor=schedule[v][use];
    if(left<1||donor<=1)continue;
    int give=reserve[v]>1?donor/2:donor-1;
    if(donor-give<reserve[v])continue;
    if(left+give>schedule[u][b]){schedule[u][b]=left+give;schedulePick[u][k][b]=use;}
   }
  }
 }
 int best=0,first=-1;
 for(int b=0;b<=budget;b++){
  if(schedule[g][b]>schedule[g][best])best=b;
  if(first<0&&schedule[g][b]>=tempoTarget)first=b;
 }
 if(plan&&first>=0){
  *plan=Program();plan->kind=6;plan->root=plan->goal=g;plan->born=tick;
  for(int j=0;j<tempoTreeCount;j++){int c=tempoNodes[j];scheduleParent[c]=tempoParent[c];}
  if(first)extractSchedule(g,first,*plan);
  plan->value=5.+.7*std::max(0,schedule[g][first]-ready(g))/(first+1.);
 }
 return schedule[g][best];
}

void prepareTempo(){
 tempoActive=false;tempoFallback=Program();tempoPin=-1;tempoUnseen=0;tempoTarget=0;tempoBudget=0;
 if(!economyReliable||economyMargin<=0||o->players!=2||aliveCount!=2||baselineDanger>0)return;
 tempoEnemy=1-p;
 if(!homeKnownTo[tempoEnemy])return;
 int visibleArmy=0,visibleLand=0;
 for(int c=0;c<N;c++)if(o->visible[c]&&own[c]==tempoEnemy){
  visibleArmy+=army(c);visibleLand++;
  if(gd[c]<=6&&army(c)>=std::max(20,ready(g)/2)&&
     (tempoPin<0||army(c)>army(tempoPin)))tempoPin=c;
 }
 if(tempoPin<0)return;
 int unseenLand=std::max<int64_t>(0,o->total_land[tempoEnemy]-visibleLand);
 tempoUnseen=int(std::max<int64_t>(0,o->total_army[tempoEnemy]-visibleArmy-unseenLand));
 int arrival=INF;
 for(int c=0;c<N;c++)if(!o->visible[c]&&ter[c]!=1)arrival=std::min(arrival,gd[c]);
 // A fog obstacle may be an enemy city, so it remains a possible source.
 // No claim is made that this pessimistic location is actually occupied.
 if(tempoUnseen<8||arrival>=INF)return;
 tempoBudget=std::clamp(arrival-int(tempoEnemy<p),1,12);
 int known=std::max(ready(tempoPin),opponentAssembly[tempoPin]);
 for(int d=1;d<=8;d++)known=std::max(known,homeIncoming[d]+futureGrowth(d));
 tempoTarget=int(std::min<int64_t>(o->total_army[tempoEnemy],known+tempoUnseen));
 if(ready(g)>=tempoTarget)return;
 if(tempoCollection(nullptr,tempoBudget,&tempoFallback)<tempoTarget)return;
 if(tempoFallback.count&&legalOrder(tempoFallback.orders[0],true))tempoActive=true;
}

bool tempoSelection(Program &chosen){
 if(!tempoActive)return false;
 int indices[18];for(int j=0;j<proposalCount;j++)indices[j]=j;
 std::sort(indices,indices+proposalCount,[&](int a,int b){return proposalScore[a]>proposalScore[b];});
 int checked=0;
 for(int j=0;j<proposalCount;j++){
  int i=indices[j];if(proposalScore[i]<=-.01-baselineDanger)break;
  const auto &q=proposals[i];auto x=q.orders[q.cursor];
  if(q.kind==4){chosen=q;return true;}
  if(tempoCollection(&x,tempoBudget-1,nullptr)>=tempoTarget){chosen=q;return true;}
  if(++checked==4)break;
 }
 Order wait;
 if(tempoCollection(&wait,tempoBudget-1,nullptr)>=tempoTarget){chosen=Program();return true;}
 if(!tempoFallback.count||!legalOrder(tempoFallback.orders[0],true))return false;
 chosen=tempoFallback;return true;
}
