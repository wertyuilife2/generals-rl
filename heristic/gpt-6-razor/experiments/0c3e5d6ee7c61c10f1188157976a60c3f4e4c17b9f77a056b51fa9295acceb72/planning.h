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
int proposalCount=0,missionTick=-1;
int missionReceiptCell=-1,missionReceiptForce=0;
std::int64_t missionAction=0;
int supplyParent[MAXN],supplyArmy[MAXN],supplyLand[MAXN];
int schedule[MAXN][13];
signed char schedulePick[MAXN][4][13];
int scheduleParent[MAXN],scheduleOrder[MAXN],scheduleCount=0;
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
Program oneStep(const Candidate &c,int kind=0)const {
 Program q;q.born=tick;q.kind=kind;q.value=c.score;q.goal=c.target;q.root=c.source;
 if(c.source>=0&&c.dir>=0){q.count=1;q.orders[0]={c.source,nb[c.source][c.dir],c.half};}
 return q;
}
void offer(const Program &q){
 if(q.cursor>=q.count||!legalOrder(q.orders[q.cursor],q.kind==3))return;
 const Order &x=q.orders[q.cursor];
 for(int i=0;i<proposalCount;i++){
  const Order &y=proposals[i].orders[proposals[i].cursor];
  if(x.s==y.s&&x.t==y.t&&x.half==y.half){if(q.value>proposals[i].value)proposals[i]=q;return;}
 }
 if(proposalCount<18){proposals[proposalCount++]=q;return;}
 int worst=0;for(int i=1;i<18;i++)if(proposals[i].value<proposals[worst].value)worst=i;
 if(q.value>proposals[worst].value)proposals[worst]=q;
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
 int local=0;
 for(int k=0;k<enemyCount[c];k++)local=std::max(local,moveArmy(enemyNeighbors[c][k]));
 double holding=std::min(80.,16.+8.*ed[c]);
 if(local>remaining+3)holding=std::min(holding,8.);
 else if(remaining>local+12)holding=std::min(90.,holding+16.);
 int exits=0;for(int k=0;k<4;k++){int v=nb[c][k];if(v>=0&&!blocked(v)&&st[v]!=1)exits++;}
 // Income is discounted by the time until occupation. The garrison is an
 // opportunity cost, not a second casualty charge on top of capture loss.
 double value=(5.+holding*.29+exits*.55)/(1+.012*arrival);
 if(own[c]>=0&&own[c]!=p)value+=6.;
 if(remaining<4)value*=.70;
 return value;
}

// Compute the exact set of edges used by a bounded friendly collection tree.
// Each child subtree is disjoint, each selected edge costs one order, and the
// reconstruction merges children before moving their parent toward the root.
void scheduleTree(int root,int budget){
 std::fill(scheduleParent,scheduleParent+N,-2);
 scheduleCount=1;scheduleOrder[0]=root;scheduleParent[root]=-1;dist[root]=0;
 for(int j=0;j<scheduleCount;j++){
  int u=scheduleOrder[j];if(dist[u]>=budget)continue;
  for(int k=0;k<4;k++){
   int v=nb[u][k];if(v<0||!mine(v)||scheduleParent[v]!=-2)continue;
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
    int value=schedule[u][b-use-1]+schedule[child][use]-1;
    if(value>schedule[u][b]){schedule[u][b]=value;schedulePick[u][k][b]=use;}
   }
  }
 }
}
void extractSchedule(int u,int budget,Program &q){
 int use[4];
 for(int k=3;k>=0;k--){use[k]=schedulePick[u][k][budget];if(use[k]>=0)budget-=use[k]+1;}
 // Near children first: they provide a useful intermediate fallback if a
 // new threat interrupts the larger program before its last merge.
 for(int k=0;k<4;k++)if(use[k]>=0){
  int child=nb[u][k];extractSchedule(child,use[k],q);
  if(q.count<40)q.orders[q.count++]={child,u,false};
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
 // Strong nearby sources first under the fixed work bound. All owners are
 // represented before additional sources, preventing player-count cutoffs.
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
    if(king)score+=6./(1+.07*d);
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
    if(king)score+=6./(1+.07*(k+d));
    if(score>value){value=score;goal=v;}
   }
   // A mobile column at a real front is a usable end state even before the
   // enemy general is known. Scale by the force it still needs, not total army.
   if(root!=g&&ed[root]<=5){
    int need=20;
    for(int c=0;c<N;c++)if(o->visible[c]&&own[c]>=0&&own[c]!=p&&
       std::abs(cellX[c]-cellX[root])+std::abs(cellY[c]-cellY[root])<=8)need=std::max(need,army(c)+12);
    int useful=std::min(gained,std::max(0,need-ready(root)));
    double score=.85*std::min(useful,ready(root)+gained/2)/(k+2.0)/(1+.07*ed[root]);
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

void weightedProgram(double income){
 if(aliveCount!=2||sources.empty())return;
 int s=sources[0];
 for(int split=0;split<=(usefulCityHalf(s)?1:0);split++){
  Candidate c=weightedCampaign(s,income,split);if(c.source<0)continue;
  Program q;q.value=c.score;q.goal=c.target;q.root=s;q.born=tick;q.kind=1;
  int path[33],n=0;for(int v=c.target;v!=s&&v>=0&&n<33;v=weightedParent[v])path[n++]=v;
  int u=s;for(int j=n-1;j>=0;j--){q.orders[q.count++]={u,path[j],j==n-1&&c.half};u=path[j];}
  offer(q);
 }
}

void readinessProgram(){
 if(ed[g]>5||homeIncoming[8]<8||baselineDanger>0)return;
 int target=std::min(int(o->total_army[p]*.45),int(homeIncoming[8]*1.35)+6);
 int deficit=target-ready(g);if(deficit<4)return;
 scheduleTree(g,8);Program best;
 for(int k=1;k<=8;k++){
  int gain=schedule[g][k]-ready(g);if(gain<4)continue;
  double score=2.1+.30*std::min(gain,deficit)/(k+1.);
  if(score<=best.value)continue;
  Program q;q.value=score;q.goal=g;q.root=g;q.kind=6;q.born=tick;extractSchedule(g,k,q);
  if(q.count&&legalOrder(q.orders[0]))best=q;
 }
 offer(best);
}

std::int64_t decideV6(const ArenaObservation *obs){
 if(obs->width*obs->height>MAXN||obs->general_pos<0)return 0;
 update(obs);supplyMap();buildAttacks();baselineDanger=dangerAfter(nullptr,homeIncoming);proposalCount=0;
 // Finish a concrete, validated program unless new information or a better
 // opportunity justifies interruption. Skipped online ticks invalidate it.
 bool received=o->previous?(o->previous->valid&&o->previous->action==missionAction):
    (missionReceiptCell>=0&&mine(missionReceiptCell)&&army(missionReceiptCell)>=missionReceiptForce);
 bool stillNeeded=(mission.kind!=3||baselineDanger>0||raidNeed>army(g))&&(mission.kind!=6||army(g)<std::min(int(o->total_army[p]*.45),int(homeIncoming[8]*1.35)+6));
 if(missionTick==tick-1&&mission.cursor<mission.count&&tick-mission.born<=32&&received&&stillNeeded&&
    legalOrder(mission.orders[mission.cursor],mission.kind==3)){
  Program continued=mission;continued.value*=1.09;offer(continued);
 }
 for(int s=0;s<N;s++)if(mine(s))for(int k=0;k<4;k++){
  int v=nb[s][k];if(v<0||st[v]!=2||own[v]<0||own[v]==p)continue;
  for(int half=0;half<=1;half++)if(moveArmy(s,half)>captureDefense(v)&&
      (aliveCount<=2||captureSafe(s,v,moveArmy(s,half)))){
   offer(oneStep({100000.+moveArmy(s,half)*.001,s,v,k,bool(half)},4));
  }
 }
 Candidate defense=defend(),forecast=forecastDefend();
 if(forecast.score>defense.score)defense=forecast;
 if(defense.source>=0)offer(oneStep(defense,3));
 // Local interception is generated by defend(); economy is not displaced
 // merely because an adjacent enemy tile can be traded.
 sources.clear();
 for(int c=0;c<N;c++)if(mine(c)&&canMoveNow(c))sources.push_back(c);
 std::sort(sources.begin(),sources.end(),[&](int x,int y){return army(x)+(x==oldSource?5:0)>army(y)+(y==oldSource?5:0);});
 if(sources.size()>10){
  keep.assign(sources.begin(),sources.begin()+6);border.clear();
  for(int j=6;j<int(sources.size());j++){
   int c=sources[j],key=ed[c];
   if(key>=INF)key=INF-unknown[c]*2;
   border.emplace_back(key,j);
  }
  std::partial_sort(border.begin(),border.begin()+4,border.end());
  for(int j=0;j<4;j++)keep.push_back(sources[border[j].second]);sources.swap(keep);
 }
 double income=.5*(std::max<int64_t>(0,o->total_cities[p])+1)+.02*o->total_land[p];
 campaignPrograms(income);
 weightedProgram(income);
 routePlan(income); // Registers the full column program, including collections.
 if(tick>=48)logisticsPrograms(income);
 readinessProgram();
 // A common post-action safety gate covers every mode and every player count.
 // Safe alternatives are preferred before comparing economic benefits.
 double bestScore=-.01-baselineDanger;int best=-1;
 for(int j=0;j<proposalCount;j++){
  Program &q=proposals[j];const Order &x=q.orders[q.cursor];
  if(!legalOrder(x,q.kind==3))continue;
  double after=(q.kind==4&&aliveCount<=2)?0:dangerAfter(&x);
  if(after>baselineDanger+1&&after>60&&q.kind!=3)continue;
  double score=q.value-after;
  if(score>bestScore){bestScore=score;best=j;}
 }
 if(best<0){mission=Program();missionTick=tick;return 0;}
 Program chosen=proposals[best];Order action=chosen.orders[chosen.cursor++];
 // Continuation preference is applied afresh, never compounded every turn.
 if(chosen.born<tick)chosen.value/=1.09;
 mission=chosen;missionTick=tick;oldTarget=chosen.goal;oldSource=action.t;lastTarget=action.t;
 missionReceiptCell=action.t;
 missionReceiptForce=moveArmy(action.s,action.half)+(mine(action.t)?army(action.t):-rememberedDefense(action.t));
 missionAction=arena_move(action.s,direction(action.s,action.t),action.half);
 return missionAction;
}
