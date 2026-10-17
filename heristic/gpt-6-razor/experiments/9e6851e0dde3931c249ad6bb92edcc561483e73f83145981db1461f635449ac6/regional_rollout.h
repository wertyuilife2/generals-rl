// Joint command-budget evaluation for a visible invasion of a productive
// region. This is a bounded response model, never a hidden-board simulator.
static constexpr int RH=16, RA=12, RC=18;
struct RegionState {
 int owner[MAXN],army[MAXN];
 bool revealed[MAXN];
 int hero=-1,kingLost=0,enemyKingLost=0;
 double discovery=0;
};
RegionState regionState;
int regionAssets[RA],regionAssetCount=0,regionEnemy=-1;
short regionParent[RA][2][MAXN],regionDistance[RA][MAXN];
Program regionChoices[RC];
int regionChoiceCount=0,regionChecked=0,regionSelected=-1;
double regionScores[RC],regionBaseScore=0;
bool regionActive=false;

bool regionalTimeLeft()const {
 return std::chrono::steady_clock::now()-planningStart<std::chrono::microseconds(4200);
}

void regionalReset(){
 auto &s=regionState;s.hero=-1;s.kingLost=s.enemyKingLost=0;s.discovery=0;
 for(int c=0;c<N;c++){
  // Only currently visible enemy troops are modeled as present. Unseen
  // land remains an uncertain obstruction; it is not assigned hidden units.
  s.owner[c]=mine(c)?p:o->visible[c]?own[c]:-3;
  s.army[c]=mine(c)?ready(c):o->visible[c]?rememberedDefense(c):0;
  s.revealed[c]=seen[c]>-1000;
 }
}

bool regionalMove(const Order &x,int player){
 auto &s=regionState;
 if(x.s<0||x.t<0||s.owner[x.s]!=player||blocked(x.t)||direction(x.s,x.t)<0)return false;
 int sent=x.half?s.army[x.s]/2:s.army[x.s]-1;if(sent<=0)return false;
 s.army[x.s]-=sent;
 if(s.owner[x.t]==player)s.army[x.t]+=sent;
 else if(sent>s.army[x.t]){
  if(x.t==g&&player!=p)s.kingLost=1;
  if(st[x.t]==2&&own[x.t]==regionEnemy&&player==p)s.enemyKingLost=1;
  s.owner[x.t]=player;s.army[x.t]=sent-s.army[x.t];
 }else s.army[x.t]-=sent;
 if(player==p&&s.owner[x.t]==p){
  s.hero=x.t;
  for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){
   int y=cellY[x.t]+dy,z=cellX[x.t]+dx;
   if(y<0||z<0||y>=H||z>=W)continue;int c=y*W+z;
   if(!s.revealed[c]){s.revealed[c]=true;s.discovery+=1.;}
  }
 }
 return true;
}

void regionalSources(int player,int *cells,int &count)const {
 count=0;const auto &s=regionState;
 for(int c=0;c<N;c++)if(s.owner[c]==player&&s.army[c]>1){
  int j=count;if(j==4){j=3;if(s.army[c]<=s.army[cells[j]])continue;}else count++;
  while(j>0&&s.army[c]>s.army[cells[j-1]]){if(j<4)cells[j]=cells[j-1];j--;}
  cells[j]=c;
 }
}

// Project a path on the CURRENT shared ledger. A blocked old approach can
// become feasible after a withdrawal; a captured donor is no longer usable.
int regionalArrival(int source,int ai,int rotation,int player,int &firstStep,int &steps)const {
 const auto &s=regionState;int c=source,n=s.army[c];firstStep=-1;steps=0;
 if(regionDistance[ai][c]>RH)return -INF;
 while(c!=regionAssets[ai]&&steps<RH){
  int v=regionParent[ai][rotation][c];if(v<0)return -INF;
  if(firstStep<0)firstStep=v;
  n-=1;if(s.owner[v]==player)n+=s.army[v];else n-=s.army[v];
  if(n<=0)return -INF;
  c=v;steps++;
 }
 return c==regionAssets[ai]?n:-INF;
}

Order regionalEnemyOrder(int mode){
 auto &s=regionState;int sources[4],count=0;regionalSources(regionEnemy,sources,count);
 Order best;double score=-1;
 for(int i=0;i<count;i++){
  int u=sources[i];
  // Capturing an exposed mobile army can matter more than taking the next
  // empty city. The remaining defender is removed on this same ledger.
  for(int k=0;k<4;k++){
   int v=nb[u][k];if(v<0||s.owner[v]!=p||s.army[u]-1<=s.army[v])continue;
   double value=(v==g?100000.:st[v]==1?32.:3.+.35*s.army[v]);
   if(value>score){score=value;best={u,v,false};}
  }
  for(int ai=0;ai<regionAssetCount;ai++){
   int goal=regionAssets[ai];if(s.owner[goal]!=p||goal==u)continue;
   for(int rotation=0;rotation<2;rotation++){
    int v,steps;int left=regionalArrival(u,ai,(rotation+mode)%2,regionEnemy,v,steps);
    if(left<1||v<0)continue;
    double prize=goal==g?(mode?2000.:400.):(mode?24.:48.);
    double value=prize/(steps+1.)+.02*std::min(100,left);
    if(value>score){score=value;best={u,v,false};}
   }
  }
 }
 if(best.s>=0)return best;
 // No funded capture: combine observed adjacent donors rather than
 // pretending the army disappears or repeatedly suicides into a garrison.
 for(int i=0;i<count;i++)for(int k=0;k<4;k++){
  int v=nb[sources[i]][k];if(v<0||s.owner[v]!=regionEnemy||s.army[v]<=1)continue;
  int u=sources[i];if(s.army[u]<s.army[v])continue;
  double value=s.army[v]-1;
  if(value>score){score=value;best={v,u,false};}
 }
 return best;
}

Order regionalCoverOrder(){
 auto &s=regionState;int root=s.hero;
 if(root<0||s.owner[root]!=p||s.army[root]<=1)return {};
 Order best;double score=0;
 for(int k=0;k<4;k++){
  int v=nb[root][k];if(v<0||s.owner[v]!=regionEnemy||s.army[root]-1<=s.army[v])continue;
  // Intercept a real column, not a one-troop distraction beside the post.
  if(s.army[v]>=8){double value=20.+s.army[v];if(value>score){score=value;best={root,v,false};}}
 }
 if(best.s>=0)return best;
 int enemies[4],count=0;regionalSources(regionEnemy,enemies,count);
 for(int ai=0;ai<regionAssetCount;ai++){
  int goal=regionAssets[ai];if(s.owner[goal]!=p)continue;
  int arrival=INF,incoming=0;
  for(int i=0;i<count;i++)for(int rotation=0;rotation<2;rotation++){
   int first,steps,left=regionalArrival(enemies[i],ai,rotation,regionEnemy,first,steps);
   if(left<=0)continue;
   if(steps<arrival){arrival=steps;incoming=left+s.army[goal];}
   else if(steps==arrival)incoming=std::max(incoming,left+s.army[goal]);
  }
  if(arrival==INF||root==goal||regionDistance[ai][root]>arrival)continue;
  for(int rotation=0;rotation<2;rotation++){
   int first,steps,left=regionalArrival(root,ai,rotation,p,first,steps);
   if(first<0||s.owner[first]!=p||left<incoming+1)continue;
   double value=(goal==g?1000.:60.)/(arrival+1.);
   if(value>score){score=value;best={root,first,false};}
  }
 }
 if(best.s>=0)return best;
 // Grow the same mobile reserve with one actual adjacent transfer.
 for(int k=0;k<4;k++){
  int v=nb[root][k];if(v<0||v==g||s.owner[v]!=p||s.army[v]<=2)continue;
  bool half=reserve[v]>1;int sent=half?s.army[v]/2:s.army[v]-1;
  if(s.army[v]-sent<reserve[v])continue;
  if(sent>score){score=sent;best={v,root,half};}
 }
 return best;
}

double regionalValue()const {
 const auto &s=regionState;if(s.kingLost)return -1000000.;if(s.enemyKingLost)return 1000000.;
 double value=.3*s.discovery;
 for(int c=0;c<N;c++){
  int sign=s.owner[c]==p?1:s.owner[c]==regionEnemy?-1:0;
  int old=mine(c)?1:o->visible[c]&&own[c]==regionEnemy?-1:0;
  int original=mine(c)?ready(c):o->visible[c]?rememberedDefense(c):0;
  value+=sign*s.army[c]-old*original;
  // Real growth has already been simulated. This is a bounded terminal
  // production value, not a new capture prize on every recapture.
  value+=(sign-old)*(st[c]==1?16.:st[c]==2?24.:.5);
 }
 return value;
}

double regionalOutcome(const Program &q,int mode,Program *continuation=nullptr){
 regionalReset();auto &s=regionState;
 s.hero=q.root>=0&&mine(q.root)?q.root:-1;
 if(continuation){*continuation=Program();continuation->kind=29;continuation->root=q.root;continuation->goal=q.goal;continuation->born=tick;}
 bool executing=true;
 for(int step=0;step<RH;step++){
  if(step)for(int c=0;c<N;c++)if(s.owner[c]>=0)s.army[c]+=grows(c,actionTurn(step+1));
  auto ours=[&](){
   Order x;
   if(executing&&q.cursor+step<q.count)x=q.orders[q.cursor+step];
   else if(step>0)x=regionalCoverOrder();
   if(x.s<0)return;
   int sent=x.half?s.army[x.s]/2:s.army[x.s]-1;
   // Never borrow the same home/asset reserve for this region and another
   // promise. A failed or intercepted order aborts the planned suffix.
   if(s.owner[x.s]!=p||sent<=0||s.army[x.s]-sent<reserve[x.s]){executing=false;return;}
   bool moved=regionalMove(x,p);
   if(!moved)executing=false;
   if(continuation&&moved&&continuation->count<40)continuation->orders[continuation->count++]=x;
  };
  auto theirs=[&](){regionalMove(regionalEnemyOrder(mode),regionEnemy);};
  if(regionEnemy<p){theirs();ours();}else{ours();theirs();}
  if(s.kingLost||s.enemyKingLost)break;
 }
 return regionalValue();
}

bool prepareRegional(){
 regionActive=false;regionChoiceCount=regionChecked=0;regionSelected=-1;regionAssetCount=0;
 if(o->players!=2||aliveCount!=2||tick<150||o->total_cities[p]<1||baselineDanger>0)return false;
 regionEnemy=1-p;int closest[MAXN];std::fill(closest,closest+N,INF);
 int threats[4],tc=0;
 for(int c=0;c<N;c++)if(o->visible[c]&&own[c]==regionEnemy&&army(c)>=20){
  int j=tc;if(j==4){j=3;if(army(c)<=army(threats[j]))continue;}else tc++;
  while(j>0&&army(c)>army(threats[j-1])){if(j<4)threats[j]=threats[j-1];j--;}
  threats[j]=c;
 }
 if(!tc)return false;
 int h=0,t=0;for(int j=0;j<tc;j++){int c=threats[j];closest[c]=0;queue[t++]=c;}
 while(h<t){int u=queue[h++];if(closest[u]>=RH)continue;
  for(int k=0;k<4;k++){int v=nb[u][k];if(v<0||blocked(v)||closest[v]<INF)continue;
   closest[v]=closest[u]+1;queue[t++]=v;}
 }
 int cities[MAXN],count=0;
 for(int c=0;c<N;c++)if(mine(c)&&st[c]==1&&closest[c]<=12)cities[count++]=c;
 if(!count)return false;
 std::sort(cities,cities+count,[&](int a,int b){return closest[a]<closest[b];});
 regionAssets[regionAssetCount++]=g;
 for(int j=0;j<count&&regionAssetCount<8;j++)regionAssets[regionAssetCount++]=cities[j];
 int extra[MAXN],ec=0;
 for(int c=0;c<N;c++)if(o->visible[c]&&!mine(c)&&st[c]==1&&closest[c]<=RH)extra[ec++]=c;
 std::sort(extra,extra+ec,[&](int a,int b){return closest[a]<closest[b];});
 for(int j=0;j<ec&&regionAssetCount<RA;j++)regionAssets[regionAssetCount++]=extra[j];
 for(int ai=0;ai<regionAssetCount;ai++)for(int rotation=0;rotation<2;rotation++){
  auto &parent=regionParent[ai][rotation];std::fill(parent,parent+N,short(-1));
  auto &distance=regionDistance[ai];std::fill(distance,distance+N,short(10000));
  int root=regionAssets[ai];h=0;t=1;queue[0]=root;distance[root]=0;
  while(h<t){int u=queue[h++];if(distance[u]>=RH)continue;
   for(int k=0;k<4;k++){int v=nb[u][(k+2*rotation)%4];if(v<0||blocked(v)||distance[v]<10000)continue;
    parent[v]=short(u);distance[v]=distance[u]+1;queue[t++]=v;}
  }
 }
 regionActive=true;return true;
}

void regionalAdd(const Program &q){
 if(regionChoiceCount==RC||q.count-q.cursor>RH||q.cursor>=q.count)return;
 auto x=q.orders[q.cursor];if(!permitted(x))return;
 for(int j=0;j<regionChoiceCount;j++){
  auto a=regionChoices[j].orders[regionChoices[j].cursor];
  if(x.s==a.s&&x.t==a.t&&x.half==a.half&&q.root==regionChoices[j].root)return;
 }
 regionChoices[regionChoiceCount++]=q;
}

void regionalSelection(Program &chosen){
 if(chosen.kind==4||chosen.kind==3||!prepareRegional())return;
 // The auction winner is always the control. Programs are compared by
 // their resulting shared state here, not by unrelated raw module prizes.
 regionChoices[regionChoiceCount++]=chosen;
 int rank[18];for(int j=0;j<proposalCount;j++)rank[j]=j;
 std::sort(rank,rank+proposalCount,[&](int a,int b){return proposalScore[a]>proposalScore[b];});
 for(int j=0;j<proposalCount&&regionChoiceCount<5;j++)regionalAdd(proposals[rank[j]]);
 for(int ai=1;ai<regionAssetCount&&regionChoiceCount<RC-1;ai++){
  if(!regionalTimeLeft())break;
  int root=regionAssets[ai];if(!mine(root))continue;
  scheduleTree(root,12);
  int need=std::max(20,zoneThreat[root]+6),best=-1;double utility=-1;
  for(int k=1;k<=12;k++)if(schedule[root][k]>ready(root)+5){
   double gain=std::min(need,schedule[root][k])-std::min(need,ready(root));
   double value=gain/(k+2.);if(value>utility){utility=value;best=k;}
  }
  for(int k:{best,12})if(k>0&&schedule[root][k]>ready(root)+5){
   Program q;q.kind=29;q.root=q.goal=root;q.born=tick;extractSchedule(root,k,q);
   regionalAdd(q);
  }
 }
 // Reassess existing mobile posts, including the turn after collection
 // finishes. A post is not erased merely because its assembly program ended.
 regionalReset();int posts[4],pc=0;regionalSources(p,posts,pc);
 for(int j=0;j<pc&&regionChoiceCount<RC-1;j++){
  regionalReset();regionState.hero=posts[j];Order x=regionalCoverOrder();
  Program q;q.kind=29;q.root=posts[j];q.goal=posts[j];q.born=tick;
  if(x.s>=0){q.count=1;q.orders[0]=x;regionalAdd(q);}
  else regionChoices[regionChoiceCount++]=q;
 }
 // A voluntary pause is also a real one-command choice. It must be scored
 // after allowing the opponent to act; no free own moves precede it.
 Program wait;wait.kind=29;wait.root=chosen.root;wait.born=tick;
 if(regionChoiceCount<RC)regionChoices[regionChoiceCount++]=wait;
 double best=-1e30;int selected=-1;
 for(int j=0;j<regionChoiceCount;j++){
  if(j>0&&!regionalTimeLeft())break;
  double first=regionalOutcome(regionChoices[j],0),second=regionalOutcome(regionChoices[j],1);
  double value=std::min(first,second);regionScores[j]=value;regionChecked++;
  if(j==0)regionBaseScore=value;
  // A tiny tie preference keeps the current meaningful program stable.
  if(j==0)value+=1.;
  if(value>best){best=value;selected=j;}
 }
 regionSelected=selected;
 if(selected<=0)return;
 auto q=regionChoices[selected];
 if(q.count==0){chosen=q;return;}
 if(q.kind==29){q.value=5.+std::max(0.,best-regionBaseScore)/(q.count-q.cursor+2.);}
 chosen=q;
}
