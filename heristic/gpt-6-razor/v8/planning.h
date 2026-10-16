// Included inside Policy. Every array and task belongs to one seat instance.
// A plan is an ordered program, rather than a promise to collect the same
// soldiers independently from several sources.
struct Order { int s=-1,t=-1; bool half=false; };
struct Program {
 double value=-1e30;
 int goal=-1,root=-1,kind=0,count=0,cursor=0,born=0;
 Order orders[40];
};
Program proposals[18],mission;
double proposalScore[18]{};
int proposalCount=0,missionTick=-1;
int missionReceiptCell=-1,missionReceiptForce=0;
std::int64_t missionAction=0;
int supplyParent[MAXN],supplyArmy[MAXN],supplyLand[MAXN];
int schedule[MAXN][25];
signed char schedulePick[MAXN][4][25];
int scheduleParent[MAXN],scheduleOrder[MAXN],scheduleMass[MAXN],scheduleCount=0;
struct AttackRoute { short cell[9]; unsigned char length=0,owner=0; int strength=-INF; };
AttackRoute attacks[8][9][5];
int attackCounts[8][9]{},attackNodes=0;
int attackPath[9];
double baselineDanger=0;
int homeIncoming[9]{};

int direction(int s,int t)const {
 for(int k=0;k<4;k++)if(nb[s][k]==t)return k;
 return -1;
}
int ready(int c)const { return moveArmy(c)+1; }
int grows(int c,int at)const {
 return int((st[c]>0&&st[c]<3&&at%2==0)||(st[c]==0&&at%50==0));
}
bool legalOrder(const Order &x,bool allowAttrition=false)const {
 if(x.s<0||x.t<0||!mine(x.s)||blocked(x.t)||direction(x.s,x.t)<0)return false;
 int sent=moveArmy(x.s,x.half);
 if(sent<=0)return false;
 if(!mine(x.t)&&sent<=rememberedDefense(x.t)&&!allowAttrition)return false;
 // A source that an earlier seat can remove is not a reliable donor.
 for(int k=0;k<enemyCount[x.s];k++){
  int e=enemyNeighbors[x.s][k];
  if(own[e]<p&&moveArmy(e)>ready(x.s))return false;
 }
 return true;
}
// Marginal concentration only: transporting an already large stack into a
// one-troop root creates no new field army. Shared branches are counted once.
// Recently split troops retain a common identity while moving. Reuniting
// that identity is transport, not newly assembled fighting strength.
int flowId[MAXN]{},flowAt[MAXN]{},flowSerial=0;
int flowS=-1,flowT=-1,flowTick=-1,flowFamily=0,flowBorn=0,flowTargetArmy=0,flowExpected=0;
bool flowHalf=false;
std::int64_t flowAction=0;
int currentFlow(int c)const {return flowId[c]&&tick-flowAt[c]<=32?flowId[c]:0;}
bool sameFlow(int a,int b)const {int f=currentFlow(a);return f&&f==currentFlow(b);}
void updateFlows(){
 for(int c=0;c<N;c++)if(!mine(c))flowId[c]=0;
 if(flowS<0||flowTick!=tick-1||!mine(flowT))return;
 bool received=o->previous?(o->previous->valid&&o->previous->action==flowAction):army(flowT)>=flowExpected;
 if(!received)return;
 int moved=o->previous?int(o->previous->moved):flowExpected-flowTargetArmy;
 if(moved<=0)return;
 int family=flowFamily,born=flowBorn;
 if(flowHalf&&!family){family=++flowSerial;born=tick;}
 if(flowHalf&&mine(flowS)){flowId[flowS]=family;flowAt[flowS]=born;}
 if(moved>=flowTargetArmy){flowId[flowT]=family;flowAt[flowT]=born;}
}
void recordFlow(const Order &x,std::int64_t action){
 flowS=x.s;flowT=x.t;flowTick=tick;flowHalf=x.half;flowAction=action;
 flowFamily=currentFlow(x.s);flowBorn=flowAt[x.s];
 flowTargetArmy=mine(x.t)?ready(x.t):0;
 flowExpected=moveArmy(x.s,x.half)+(mine(x.t)?army(x.t):-rememberedDefense(x.t));
}
int assemblyCredit(const Program &collection,int root,int finalForce,int need)const {
 int cells[41],families[41],mass[41]{},count=0;
 cells[0]=root;for(int j=0;j<collection.count;j++)cells[j+1]=collection.orders[j].s;
 for(int j=0;j<=collection.count;j++){
  int c=cells[j],family=currentFlow(c);if(!family)family=-c-1;
  int slot=0;while(slot<count&&families[slot]!=family)slot++;
  if(slot==count)families[count++]=family;
  mass[slot]+=std::max(0,ready(c)-1);
 }
 int largest=1;for(int j=0;j<count;j++)largest=std::max(largest,mass[j]+1);
 return std::min(std::max(0,finalForce-largest),std::max(0,need-largest));
}
Program oneStep(const Candidate &c,int kind=0)const {
 Program q;q.born=tick;q.kind=kind;q.value=c.score;q.goal=c.target;q.root=c.source;
 if(c.source>=0&&c.dir>=0){q.count=1;q.orders[0]={c.source,nb[c.source][c.dir],c.half};}
 return q;
}
// Prune proposals by the same utility that selects the actual order. Raw
// campaign prizes cannot displace a safer alternative before its safety cost
// has been evaluated; this also applies when two plans share their first move.
double programScore(const Program &q)const {
 if(q.cursor>=q.count)return -1e30;
 const Order &x=q.orders[q.cursor];
 if(!legalOrder(x,q.kind==3))return -1e30;
 if(q.kind==14)return extractionSafe(q)?q.value+(q.born<tick?40.:0):-1e30;
 if(q.kind==10&&zoneETA[x.t]<=2&&zoneThreat[x.t]>moveArmy(x.s,x.half)+(mine(x.t)?army(x.t):-rememberedDefense(x.t))+5)return -1e30;
 double after=((q.kind==4&&aliveCount<=2)||q.kind==9)?0:dangerAfter(&x);
 if(after>baselineDanger+1&&after>60&&q.kind!=3)return -1e30;
 double transferRisk=(q.kind==2||q.kind==6)?risk(x.s,x.t,moveArmy(x.s,x.half)):0;
 double continuedRisk=(q.born<tick&&(q.kind==1||q.kind==8))?risk(x.s,x.t,moveArmy(x.s,x.half)):0;
 double commitment=0;
 if(q.born<tick&&q.kind==11&&baselineDanger<2&&permitted(x))commitment=1.;
 return q.value+commitment-after-transferRisk-continuedRisk;
}
void offer(Program q){
 if(q.cursor>=q.count)return;
 Order &order=q.orders[q.cursor];
 // A route may reach a city several turns after it was planned. Revalidate
 // its departure reserve on every executed order, including continuations.
 if(q.kind!=3&&q.kind!=4&&q.kind!=9&&q.kind!=14&&st[order.s]==1&&reserve[order.s]>1&&!permitted(order)){
  order.half=true;if(!permitted(order))return;
 }
 double value=programScore(q);if(value<=-1e29)return;
 const Order &x=q.orders[q.cursor];
 for(int i=0;i<proposalCount;i++){
  const Order &y=proposals[i].orders[proposals[i].cursor];
  if(x.s==y.s&&x.t==y.t&&x.half==y.half){if(value>proposalScore[i]){proposals[i]=q;proposalScore[i]=value;}return;}
 }
 if(proposalCount<18){proposalScore[proposalCount]=value;proposals[proposalCount++]=q;return;}
 int worst=0;for(int i=1;i<18;i++)if(proposalScore[i]<proposalScore[worst])worst=i;
 if(value>proposalScore[worst]){proposals[worst]=q;proposalScore[worst]=value;}
}

// Price access to the future army, including a shorter way to recover an
// existing branch. A new tile is valuable when it joins productive territory
// to a short route; mere geometric proximity does not count as connectivity.
void supplyMap(){
 std::fill(homeDistance,homeDistance+N,INF);
 std::fill(supplyParent,supplyParent+N,-1);
 int head=0,tail=1;queue[0]=g;homeDistance[g]=0;
 while(head<tail){int u=queue[head++];
  for(int k=0;k<4;k++){int v=nb[u][k];if(v<0||!mine(v)||homeDistance[v]<INF)continue;
   homeDistance[v]=homeDistance[u]+1;supplyParent[v]=u;queue[tail++]=v;}
 }
 for(int i=0;i<N;i++){supplyArmy[i]=mine(i)?std::max(0,army(i)-1):0;supplyLand[i]=mine(i);}
 for(int j=tail-1;j>0;j--){int u=queue[j],v=supplyParent[u];supplyArmy[v]+=supplyArmy[u];supplyLand[v]+=supplyLand[u];}
 for(int c=0;c<N;c++)if(!mine(c)&&!blocked(c)){
  int shortest=INF,neighbors=0;
  for(int k=0;k<4;k++){int v=nb[c][k];if(v>=0&&mine(v)){neighbors++;shortest=std::min(shortest,homeDistance[v]+1);}}
  double shortcut=0;
  if(shortest<INF)for(int k=0;k<4;k++){
   int v=nb[c][k];if(v<0||!mine(v)||homeDistance[v]>=INF)continue;
   int saved=homeDistance[v]-shortest-1;
   if(saved>0)shortcut=std::max(shortcut,std::min(6.,.10*saved*std::sqrt(double(supplyArmy[v]+supplyLand[v]))));
  }
  networkBonus[c]+=shortcut;
  // The small local term biases equal-income expansion toward recoverable
  // branches. It cannot justify buying an otherwise unprofitable city.
  if(st[c]!=1){
   int distance=shortest<INF?shortest:gd[c];
   networkBonus[c]+=.55/(1+.07*distance)+.10*std::min(2,neighbors)
                    -.025*std::min(16,std::max(0,distance-10));
  }
 }
}

double cityUtility(int c,int remaining,int arrival)const {
 return assetUtility(c,remaining,arrival);
}

// Compute the exact set of edges used by a bounded friendly collection tree.
// Each child subtree is disjoint, each selected edge costs one order, and the
// reconstruction merges children before moving their parent toward the root.
void scheduleTree(int root,int budget){
 std::fill(scheduleParent,scheduleParent+N,-2);
 for(int c=0;c<N;c++)if(mine(c)){
  scheduleMass[c]=ready(c);
  for(int k=0;k<4;k++){int v=nb[c][k];if(v>=0&&mine(v))scheduleMass[c]+=std::max(0,ready(v)-1);}
 }
 scheduleCount=1;scheduleOrder[0]=root;scheduleParent[root]=-1;dist[root]=0;
 for(int j=0;j<scheduleCount;j++){
  int u=scheduleOrder[j];if(dist[u]>=budget)continue;
  for(int k=0;k<4;k++){
   int v=nb[u][k];if(v<0||!mine(v))continue;
   if(scheduleParent[v]!=-2){
    if(dist[v]==dist[u]+1&&scheduleParent[v]>=0&&scheduleMass[u]>scheduleMass[scheduleParent[v]])scheduleParent[v]=u;
    continue;
   }
   if(v==g&&v!=root&&reserve[g]>1)continue;
   // Exposed branches are not promised to a long assembly operation.
   if(v!=root&&enemyCount[v]&&ed[root]>ed[v])continue;
   scheduleParent[v]=u;dist[v]=dist[u]+1;scheduleOrder[scheduleCount++]=v;
  }
 }
 for(int j=scheduleCount-1;j>=0;j--){
  int u=scheduleOrder[j];schedule[u][0]=ready(u);
  for(int b=1;b<=budget;b++)schedule[u][b]=-INF;
  for(int k=0;k<4;k++){
   for(int b=0;b<=budget;b++)schedulePick[u][k][b]=-1;
   int child=nb[u][k];if(child<0||scheduleParent[child]!=u)continue;
   for(int b=budget;b>=1;b--)for(int use=0;use<b;use++){
    if(schedule[u][b-use-1]<0||schedule[child][use]<=1)continue;
    int available=schedule[child][use]-1;
    if(reserve[child]>1)available=(schedule[child][use]+1)/2>=reserve[child]?schedule[child][use]/2:0;
    if(available<=0)continue;
    int value=schedule[u][b-use-1]+available;
    if(value>schedule[u][b]){schedule[u][b]=value;schedulePick[u][k][b]=use;}
   }
  }
 }
}
void extractSchedule(int u,int budget,Program &q){
 int use[4];
 for(int k=3;k>=0;k--){use[k]=schedulePick[u][k][budget];if(use[k]>=0)budget-=use[k]+1;}
 // Every selected child is finished before its edge to the parent is used.
 // This preserves a feasible prefix if an emergency interrupts the program.
 for(int k=0;k<4;k++)if(use[k]>=0){
  int child=nb[u][k];extractSchedule(child,use[k],q);
  if(q.count<40)q.orders[q.count++]={child,u,reserve[child]>1};
 }
}

void saveAttackRoute(int enemy,int depth,int strength){
 int &count=attackCounts[enemy][depth],slot=count;
 if(count==5){slot=0;for(int j=1;j<5;j++)if(attacks[enemy][depth][j].strength<attacks[enemy][depth][slot].strength)slot=j;
  if(strength<=attacks[enemy][depth][slot].strength)return;
 }else count++;
 auto &r=attacks[enemy][depth][slot];r.owner=enemy;r.length=depth;r.strength=strength;
 for(int j=0;j<=depth;j++)r.cell[j]=attackPath[j];
}
void attackWalk(int u,int enemy,int force,int depth){
 if(depth>=8||attackNodes++>=32000)return;
 int d=depth+1;
 int sent=force-1+(depth?grows(u,actionTurn(d)):0);
 for(int k=0;k<4;k++){
  int v=nb[u][k];if(v<0||blocked(v)||gd[v]>8-d)continue;
  bool repeat=false;for(int j=0;j<=depth;j++)if(attackPath[j]==v){repeat=true;break;}if(repeat)continue;
  attackPath[d]=v;
  if(v==g){saveAttackRoute(enemy,d,sent);continue;}
  int next=sent;
  if(o->visible[v]&&own[v]==enemy){
   int growth=(st[v]>0&&st[v]<3)?actionTurn(d)/2-actionTurn(1)/2:actionTurn(d)/50-actionTurn(1)/50;
   next+=ready(v)+growth;
  }else if(mine(v)||own[v]==-2)next-=mine(v)?ready(v):rememberedDefense(v);
  // Keep blocked routes as well: withdrawing their screen can enable them.
  attackWalk(v,enemy,next,d);
 }
}
void buildAttacks(){
 std::memset(attackCounts,0,sizeof(attackCounts));attackNodes=0;
 // Visit sources round-robin by owner under a fixed work bound. The same
 // threat model is used at every player count.
 int starts[8][8],counts[8]{};
 for(int c=0;c<N;c++)if(o->visible[c]&&own[c]>=0&&own[c]!=p&&gd[c]<=8&&ready(c)>2){
  int e=own[c],pos=counts[e];
  if(pos<8)counts[e]++;else {pos=7;if(ready(c)<=ready(starts[e][pos]))continue;}
  while(pos>0&&ready(c)>ready(starts[e][pos-1])){if(pos<8)starts[e][pos]=starts[e][pos-1];pos--;}
  starts[e][pos]=c;
 }
 for(int j=0;j<8;j++)for(int e=0;e<o->players;e++)if(j<counts[e]){
  int c=starts[e][j];attackPath[0]=c;attackWalk(c,e,ready(c),0);
 }
}
double dangerAfter(const Order *order,int *envelope=nullptr)const {
 int source=-1,dest=-1,sent=0,victim=-1;
 if(order){source=order->s;dest=order->t;sent=moveArmy(source,order->half);
  if(st[dest]==2&&own[dest]>=0&&own[dest]!=p&&sent>captureDefense(dest))victim=own[dest];}
 bool captured=dest>=0&&!mine(dest)&&sent>rememberedDefense(dest);
 int home=ready(g)-(source==g?sent:0)+(dest==g?sent:0);
 int arriving[8][9]{};
 for(int e=0;e<o->players;e++)if(e!=p&&e!=victim&&o->alive[e])for(int d=1;d<=8;d++)
  for(int j=0;j<attackCounts[e][d];j++){
   const auto &r=attacks[e][d][j];int force=ready(r.cell[0]);
   if(captured&&r.cell[0]==dest&&e>p)continue;
   bool dead=false;
   for(int z=1;z<=d;z++){
    int from=r.cell[z-1],c=r.cell[z];
    force-=1;if(z>1)force+=grows(from,actionTurn(z));
    if(force<=0){dead=true;break;}
    if(c==g)break;
    bool changed=!(z==1&&e<p);
    if(changed&&c==dest&&captured){force-=sent-rememberedDefense(dest);}
    else if(o->visible[c]&&own[c]==e){
     int extra=(st[c]>0&&st[c]<3)?actionTurn(z)/2-actionTurn(1)/2:actionTurn(z)/50-actionTurn(1)/50;
     force+=ready(c)+extra;
    }else if(mine(c))force-=ready(c)+(changed?((c==dest?sent:0)-(c==source?sent:0)):0);
    else if(own[c]==-2)force-=rememberedDefense(c);
    if(force<=0){dead=true;break;}
   }
   if(!dead)arriving[e][d]=std::max(arriving[e][d],force);
  }
 // Separate adjacent armies can make consecutive attrition attacks. They
 // need not first merge into one winning stack to exhaust the garrison.
 for(int e=0;e<o->players;e++)if(e!=p&&e!=victim&&o->alive[e]){
  int adjacent[4],count=0;
  for(int k=0;k<4;k++){int c=nb[g][k];if(c<0||!o->visible[c]||own[c]!=e)continue;
   if(captured&&c==dest&&e>p)continue;
   adjacent[count++]=moveArmy(c);
  }
  std::sort(adjacent,adjacent+count,std::greater<int>());int combined=0;
  for(int d=1;d<=8;d++){if(d<=count)combined+=adjacent[d-1];arriving[e][d]=std::max(arriving[e][d],combined);}
 }
 double worst=0;int previous[8]{};
 for(int d=1;d<=8;d++){
  int incoming=0;
  for(int e=0;e<o->players;e++){previous[e]=std::max(previous[e],arriving[e][d]);incoming+=previous[e];}
  int growth=actionTurn(d)/2-actionTurn(1)/2;
  if(envelope)envelope[d]=incoming-growth;
  int lack=incoming-home-growth;
  if(lack>0){double weight=d==1?200.:d==2?45.:d==3?15.:d==4?5.:2.;worst=std::max(worst,weight*lack/(d+.5));}
 }
 return worst;
}


// Visible reserves can reinforce a known general while a distant attack is
// travelling. Reduce the prize of a raid whose expected arrival loses to that
// response, without forbidding forward pressure or inventing hidden troops.
int kingRelief[8][33]{};
void buildKingRelief(){
 std::memset(kingRelief,0,sizeof(kingRelief));
 for(int e=0;e<o->players;e++)if(e!=p&&o->alive[e]&&enemyGeneral[e]>=0){
  int root=enemyGeneral[e];std::fill(dist,dist+N,INF);dist[root]=0;int h=0,z=1;queue[0]=root;
  while(h<z){int u=queue[h++];if(dist[u]>=31)continue;
   for(int k=0;k<4;k++){int v=nb[u][k];if(v<0||blocked(v)||own[v]!=e||dist[v]<INF)continue;
    dist[v]=dist[u]+1;queue[z++]=v;}
  }
  for(int c=0;c<N;c++)if(c!=root&&o->visible[c]&&own[c]==e&&dist[c]<32){
   int arrival=dist[c]+int(e>p),force=std::max(0,ready(c)-dist[c]);
   kingRelief[e][arrival]=std::max(kingRelief[e][arrival],force);
  }
  for(int d=1;d<=32;d++)kingRelief[e][d]=std::max(kingRelief[e][d],kingRelief[e][d-1]);
  // One defender can merge several visible donors before a distant attack
  // arrives. A disjoint tree counts their action costs and shared corridors.
  std::fill(scheduleParent,scheduleParent+N,-2);
  int nodes=1;scheduleOrder[0]=root;scheduleParent[root]=-1;dist[root]=0;
  for(int j=0;j<nodes;j++){
   int u=scheduleOrder[j];if(dist[u]>=16)continue;
   for(int k=0;k<4;k++){int v=nb[u][k];
    if(v<0||blocked(v)||own[v]!=e||scheduleParent[v]!=-2||nodes>=128)continue;
    scheduleParent[v]=u;dist[v]=dist[u]+1;scheduleOrder[nodes++]=v;
   }
  }
  for(int j=nodes-1;j>=0;j--){
   int u=scheduleOrder[j];schedule[u][0]=std::max(1,rememberedDefense(u));
   for(int b=1;b<=24;b++)schedule[u][b]=-INF;
   for(int k=0;k<4;k++){int v=nb[u][k];if(v<0||scheduleParent[v]!=u)continue;
    for(int b=24;b>=1;b--)for(int use=0;use<b;use++){
     if(schedule[u][b-use-1]<0||schedule[v][use]<=1)continue;
     schedule[u][b]=std::max(schedule[u][b],schedule[u][b-use-1]+schedule[v][use]-1);
    }
   }
   for(int b=1;b<=24;b++)schedule[u][b]=std::max(schedule[u][b],schedule[u][b-1]);
  }
  for(int d=1;d<=32;d++){
   int budget=std::clamp(d-int(e>p),0,24);
   kingRelief[e][d]=std::max(kingRelief[e][d],schedule[root][budget]-std::max(1,rememberedDefense(root)));
  }
 }
}
double kingConfidence(int target,int remaining,int depth)const {
 if(depth<=1||own[target]<0||st[target]!=2)return 1.;
 int relief=kingRelief[own[target]][std::min(32,depth)];
 return std::clamp(double(std::max(1,remaining))/(relief+depth/2.+3),.20,1.);
}

void campaignPrograms(double income){
 for(int s:sources){
  bool requiredHalf=false;if(!canDepart(s,requiredHalf))continue;
  int modes=(!requiredHalf&&usefulCityHalf(s))?2:1;
  for(int mode=0;mode<modes;mode++){
   bool half=requiredHalf||mode;int moving=moveArmy(s,half);if(moving<1)continue;
   search(s,moving,32);
   double highest=-1e30;int target=-1;
   for(int v=0;v<N;v++)if(!mine(v)&&dist[v]<INF&&first[v]>=0){
    int d=dist[v];bool enemy=own[v]>=0&&own[v]!=p,king=st[v]==2&&enemy,city=st[v]==1;
    if(!king&&d>24)continue;
    if(king&&force[v]<d/2+2)continue;
    if(king&&d==1&&moving<=captureDefense(v))continue;
    double reward=king?eliminationValue(own[v]):city?cityUtility(v,force[v],d):
      1.8+(enemy?2.1:0)+unknown[v]*.22+(enemy&&moving>=20&&force[v]>=8?unknown[v]*.35:0);
    if(!king)reward+=networkBonus[v];
    double score=reward/(d+.8+pathLoss[v]/(income*2.5+2));
    if(king){score+=6./(1+.07*d);score*=kingConfidence(v,force[v],d);}
    int next=nb[s][first[v]];double hazard=risk(s,next,moving);if(hazard>=1000)continue;
    score-=hazard+(king?0:pathExposure[v]);
    if(score>highest){highest=score;target=v;}
   }
   if(target<0)continue;
   Program q;q.value=highest;q.goal=target;q.root=s;q.born=tick;q.kind=1;
   int path[33],n=0;for(int v=target;v!=s&&v>=0&&n<33;v=parent[v])path[n++]=v;
   int u=s;for(int j=n-1;j>=0;j--){q.orders[q.count++]={u,path[j],j==n-1?half:false};u=path[j];}
   offer(q);
  }
 }
}

// Compare executable assembly-and-attack programs to direct campaigns. The
// approach route never credits the tree's donors a second time.
void logisticsPrograms(double income){
 int roots[8],rootCount=0;
 auto addRoot=[&](int c){if(c<0||!mine(c)||rootCount==8)return;for(int j=0;j<rootCount;j++)if(roots[j]==c)return;roots[rootCount++]=c;};
 if(baselineDanger>0||raidNeed>army(g))addRoot(g);
 for(int c:sources)if(ed[c]<=6&&ready(c)>=3){addRoot(c);if(rootCount>=4)break;}
 for(int c:sources){addRoot(c);if(rootCount>=6)break;}
 for(int ri=0;ri<rootCount;ri++){
  int root=roots[ri],budget=10;
  scheduleTree(root,budget);
  int gathered[13];for(int k=0;k<=budget;k++)gathered[k]=schedule[root][k];
  // Paths with positive movement and combat costs; no speculative collection
  // on the attack leg, because its soldiers may already be in the tree.
  std::fill(dist,dist+N,INF);std::fill(pathLoss,pathLoss+N,INF);
  int h=0,z=1;queue[0]=root;dist[root]=0;pathLoss[root]=0;parent[root]=-1;
  while(h<z){int u=queue[h++];if(dist[u]>=14)continue;
   for(int k=0;k<4;k++){
    int v=nb[u][k];if(v<0||v==root||blocked(v))continue;
    int d=dist[u]+1;if(dist[v]<d)continue;
    int loss=pathLoss[u]+1+(mine(v)?0:rememberedDefense(v)+int(ter[v]==255));
    if(dist[v]==INF){dist[v]=d;queue[z++]=v;}
    if(loss<pathLoss[v]){pathLoss[v]=loss;parent[v]=u;}
   }
  }
  Program best;
  for(int k=1;k<=budget;k++)if(gathered[k]>ready(root)+3){
   int gained=gathered[k]-ready(root);
   double value=-1e30;int goal=-1;
   if(root==g&&(baselineDanger>0||raidNeed>army(g))){
    int need=raidNeed,deadline=9;
    for(int d=1;d<=8;d++)if(homeIncoming[d]>army(g)){need=std::max(need,homeIncoming[d]);deadline=std::min(deadline,d);}
    if(k<=std::max(1,deadline-1))value=(baselineDanger>0?40.:1.)+std::min(gained,std::max(0,need-army(g)))*2./(k+.5);
    goal=g;
   }
   int available=gathered[k]-1;
   if(reserve[root]>1){available=(gathered[k]+1)/2<reserve[root]?0:gathered[k]/2;}
   for(int v=0;v<N;v++)if(!mine(v)&&dist[v]<=14&&(st[v]==1||(st[v]==2&&own[v]>=0))){
    bool king=st[v]==2;int d=dist[v];
    int growth=own[v]>=0?(k+d)/2:0;
    int remaining=available-pathLoss[v]+1-growth;
    if(remaining<(king?4:8))continue;
    if(moveArmy(root)-pathLoss[v]+1-growth>=(king?4:8))continue;
    double reward=king?eliminationValue(own[v]):cityUtility(v,remaining,k+d)+networkBonus[v];
    double score=reward/(k+d+.8+(pathLoss[v]+k)/(income*2.5+2));
    if(king){score+=6./(1+.07*(k+d));score*=kingConfidence(v,remaining,k+d);}
    if(score>value){value=score;goal=v;}
   }
   // A mobile column at a real front is a usable end state even before the
   // enemy general is known. Scale by the force it still needs, not total army.
   if(root!=g&&ed[root]<=5){
    int need=20;
    for(int c=0;c<N;c++)if(o->visible[c]&&own[c]>=0&&own[c]!=p&&
       std::abs(cellX[c]-cellX[root])+std::abs(cellY[c]-cellY[root])<=8)need=std::max(need,army(c)+12);
    Program collection;extractSchedule(root,k,collection);
    int useful=assemblyCredit(collection,root,gathered[k],need);
    double score=std::min(6.,.85*useful/(k+2.0)/(1+.07*ed[root]));
    if(score>value){value=score;goal=root;}
   }
   if(value<=best.value||goal<0)continue;
   Program q;q.value=value;q.goal=goal;q.root=root;q.born=tick;q.kind=goal==g?3:2;
   extractSchedule(root,k,q);
   if(goal!=root){
    int path[16],n=0;for(int c=goal;c!=root&&c>=0&&n<16;c=parent[c])path[n++]=c;
    int u=root;for(int j=n-1;j>=0&&q.count<40;j--){q.orders[q.count++]={u,path[j],j==n-1&&reserve[root]>1};u=path[j];}
   }
   if(q.count&&legalOrder(q.orders[0]))best=q;
  }
  offer(best);
 }
}

void readinessProgram(){
 if(ed[g]>8||homeIncoming[8]<8||baselineDanger>0)return;
 int target=std::min(int(o->total_army[p]*.45),int(homeIncoming[8]*1.35)+6);
 int deficit=target-ready(g);if(deficit<4)return;
 scheduleTree(g,8);Program best;
 for(int k=1;k<=8;k++){
  int gain=schedule[g][k]-ready(g);if(gain<4)continue;
  double score=5.0+.70*std::min(gain,deficit)/(k+1.);
  if(score<=best.value)continue;
  Program q;q.value=score;q.goal=g;q.root=g;q.kind=6;q.born=tick;extractSchedule(g,k,q);
  if(q.count&&legalOrder(q.orders[0]))best=q;
 }
 offer(best);
}


// Deny a visible staging army before it can merge into a lethal column.
// These proposals have to pass the same post-action route checks as defence.
void interceptionPrograms(){
 for(int s=0;s<N;s++)if(mine(s)&&ready(s)>3)for(int k=0;k<4;k++){
  int v=nb[s][k];if(v<0||!o->visible[v]||own[v]<0||own[v]==p||army(v)<7||gd[v]>8)continue;
  for(int h=0;h<=1;h++){
   int sent=moveArmy(s,h),guard=rememberedDefense(v);
   if(sent<=guard+1)continue;
   Order x{s,v,bool(h)};if(!legalOrder(x))continue;
   double after=dangerAfter(&x);
   // Do not trade a safe king for an attractive but exposed attack.
   if(after>baselineDanger+.1)continue;
   if(s!=g&&risk(s,v,sent)>=1000)continue;
   int residual=sent-guard,retaliation=0;
   for(int j=0;j<enemyCount[v];j++){
    int e=enemyNeighbors[v][j];if(e==s)continue;
    retaliation=std::max(retaliation,moveArmy(e));
   }
   double value=3.5+(.18*guard)/(1.+.3*gd[v]);
   if(st[v]==1)value+=4.;
   value-=std::max(0,retaliation-residual)*.025;
   if(s!=g)value-=risk(s,v,sent);
   offer(oneStep({value,s,v,k,bool(h)},7));
  }
 }
}

// Transport idle reserves to an actual hostile target. The reward depends on
// deployable force, not the number of friendly tiles crossed or re-collected.
void deploymentPrograms(){
 if(tick<100)return;
 for(int s:sources){
  bool half=false;if(!canDepart(s,half))continue;
  // Deployment transports a reserve; it does not by itself justify giving
  // away a productive city. Tactical capture programs can still use full force.
  if(usefulCityHalf(s))half=true;
  int initial=moveArmy(s,half);if(initial<18)continue;
  search(s,initial,32);double best=-1e30;int goal=-1;
  for(int v=0;v<N;v++)if(own[v]>=0&&own[v]!=p&&o->alive[own[v]]&&dist[v]<INF&&first[v]>=0&&dist[v]>=2){
   if(force[v]<10||force[v]<initial/3)continue;
   int d=dist[v],enemy=own[v];
   double mobile=std::max<int64_t>(20,o->total_army[enemy]-o->total_land[enemy]);
   double usable=std::min(double(initial),mobile);
   double reward=2.0+.10*usable;
   if(st[v]==2)reward+=12.;
   else if(st[v]==1)reward+=3.;
   if(gd[v]<=6&&army(v)>=10)reward+=std::min(10.,army(v)*.12);
   // Saturate strategic transport value without tying every large army.
   // A hard cap erased distance differences and sent rear columns first.
   double raw=reward/(1.5+.30*d+pathLoss[v]*.04);
   double score=-6.*std::expm1(-raw/6.);
   int next=nb[s][first[v]];
   score-=risk(s,next,initial)+pathExposure[v];
   if(score>best){best=score;goal=v;}
  }
  if(goal<0)continue;
  Program q;q.value=best;q.goal=goal;q.root=s;q.kind=8;q.born=tick;
  int path[33],n=0;for(int c=goal;c!=s&&c>=0&&n<33;c=parent[c])path[n++]=c;
  int u=s;for(int j=n-1;j>=0;j--){q.orders[q.count++]={u,path[j],j==n-1&&half};u=path[j];}
  offer(q);
 }
}


// In a duel, resolve plausible first replies before pricing a home attack.
// An earlier opponent hitting home meets the pre-move guard; our subsequent
// full/half departure uses the surviving army. The static gate cannot express
// that order and would forbid even a counterattack that removes the attacker.
int responseOwner[MAXN],responseArmy[MAXN];
bool counterResponseSafe(const Order &x,int enemy,int es,int et,bool eh){
 for(int c=0;c<N;c++){
  responseOwner[c]=own[c];responseArmy[c]=own[c]>=0?ready(c):rememberedDefense(c);
 }
 auto move=[&](int s,int t,bool half,int player){
  if(s<0||responseOwner[s]!=player)return;
  int sent=half?responseArmy[s]/2:std::max(0,responseArmy[s]-1);
  responseArmy[s]-=sent;
  if(responseOwner[t]==player)responseArmy[t]+=sent;
  else if(sent>responseArmy[t]){responseArmy[t]=sent-responseArmy[t];responseOwner[t]=player;}
  else responseArmy[t]-=sent;
 };
 if(enemy<p)move(es,et,eh,enemy);
 if(responseOwner[g]!=p)return false;
 move(x.s,x.t,x.half,p);
 if(enemy>p)move(es,et,eh,enemy);
 if(responseOwner[g]!=p)return false;
 // Check every retained route and suffix: the first reply may have advanced
 // its army to a square that was ours in the observation.
 for(int d=1;d<=8;d++)for(int j=0;j<attackCounts[enemy][d];j++){
  const auto &r=attacks[enemy][d][j];
  for(int from=std::max(0,d-6);from<d;from++){
   int source=r.cell[from];if(responseOwner[source]!=enemy||responseArmy[source]<=1)continue;
   int force=responseArmy[source];bool stopped=false;
   for(int z=from+1;z<=d;z++){
    int step=z-from,u=r.cell[z-1],v=r.cell[z];
    int at=actionTurn(step+1);force+=grows(u,at);force--;
    if(force<=0){stopped=true;break;}
    int extra=(st[v]>0&&st[v]<3)?at/2-actionTurn(1)/2:at/50-actionTurn(1)/50;
    int guard=responseArmy[v]+(responseOwner[v]>=0?extra:0);
    if(v==g){if(force>guard)return false;break;}
    if(responseOwner[v]==enemy)force+=guard;
    else if(responseOwner[v]==p||responseOwner[v]==-2)force-=guard;
    if(force<=0){stopped=true;break;}
   }
   (void)stopped;
  }
 }
 return true;
}
void homeCounterPrograms(){
 if(aliveCount!=2||enemyCount[g]==0)return;
 int enemy=-1;for(int e=0;e<o->players;e++)if(e!=p&&o->alive[e])enemy=e;
 // Replies are drawn from observed mobile groups, with a fixed work bound.
 int starts[12],count=0;
 for(int c=0;c<N;c++)if(o->visible[c]&&own[c]==enemy&&gd[c]<=9&&ready(c)>1){
  int pos=count;double value=double(ready(c))/(1+gd[c]);
  if(pos<12)count++;else {pos=11;if(value<=double(ready(starts[pos]))/(1+gd[starts[pos]]))continue;}
  while(pos>0&&value>double(ready(starts[pos-1]))/(1+gd[starts[pos-1]])){if(pos<12)starts[pos]=starts[pos-1];pos--;}
  starts[pos]=c;
 }
 for(int k=0;k<enemyCount[g];k++){
  int v=enemyNeighbors[g][k];if(army(v)<7||st[v]==2)continue;
  for(int h=1;h>=0;h--){
   int sent=moveArmy(g,h);if(sent<=rememberedDefense(v)+2)continue;
   Order x{g,v,bool(h)};if(!counterResponseSafe(x,enemy,-1,-1,false))continue;
   bool safe=true;
   for(int j=0;j<count&&safe;j++)for(int dir=0;dir<4&&safe;dir++){
    int e=starts[j],t=nb[e][dir];if(t<0||blocked(t))continue;
    for(int half=0;half<(ready(e)>=10?2:1);half++)if(!counterResponseSafe(x,enemy,e,t,half)){safe=false;break;}
   }
   if(safe)offer(oneStep({8.+std::min(60.,army(v)*.40),g,v,direction(g,v),bool(h)},9));
  }
 }
}

