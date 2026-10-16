// Plan reconnaissance as one executable investment: collect disjoint donors,
// then carry the resulting column through a safe approach to unseen ground.
// Route estimates deliberately do not collect those donors a second time.
int expeditionLoss[MAXN],expeditionDepth[MAXN],expeditionParent[MAXN];
double expeditionCost[MAXN];
unsigned char expeditionVision[MAXN];

void expeditionPrograms(){
 if(tick<180||o->total_land[p]<45||baselineDanger>1)return;
 bool unknownKing=false;
 for(int e=0;e<o->players;e++)if(e!=p&&o->alive[e]&&enemyGeneral[e]<0)unknownKing=true;
 if(!unknownKing)return;
 int roots[8],count=0;double priorities[8];
 for(int c=0;c<N;c++)if(mine(c)&&ready(c)>2){
  bool half=false;if(!canDepart(c,half))continue;
  int mobile=moveArmy(c,half);
  for(int k=0;k<4;k++){int v=nb[c][k];if(v>=0&&mine(v)&&canDepart(v,half))mobile+=moveArmy(v,half)/2;}
  double value=std::min(110,mobile)/(2.+.15*std::min(30,ed[c]));
  if(zoneETA[c]<=2&&zoneThreat[c]>=ready(c))continue;
  int j=count;if(j==8){j=7;if(value<=priorities[j])continue;}else count++;
  while(j>0&&value>priorities[j-1]){if(j<8){roots[j]=roots[j-1];priorities[j]=priorities[j-1];}j--;}
  roots[j]=c;priorities[j]=value;
 }
 for(int ri=0;ri<count;ri++){
  int root=roots[ri];bool requiredHalf=false;canDepart(root,requiredHalf);
  scheduleTree(root,10);
  std::fill(expeditionCost,expeditionCost+N,1e30);
  std::fill(expeditionParent,expeditionParent+N,-1);
  std::fill(expeditionLoss,expeditionLoss+N,INF);
  std::fill(expeditionDepth,expeditionDepth+N,INF);
  using Entry=std::pair<double,int>;
  std::priority_queue<Entry,std::vector<Entry>,std::greater<Entry>> heap;
  expeditionCost[root]=0;expeditionDepth[root]=0;expeditionLoss[root]=0;heap.push({0,root});
  int expanded=0;
  while(!heap.empty()&&expanded<N*2){
   auto [cost,u]=heap.top();heap.pop();if(cost>expeditionCost[u]+1e-8)continue;expanded++;
   if(expeditionDepth[u]>=24)continue;
   for(int k=0;k<4;k++){
    int v=nb[u][k];if(v<0||v==root||blocked(v))continue;
    int guard=mine(v)?0:rememberedDefense(v);
    double exposure=zoneETA[v]<=2?std::max(0,zoneThreat[v]-35)*.12:0.;
    double next=cost+1.+guard*.18+exposure;
    if(next>=expeditionCost[v])continue;
    expeditionCost[v]=next;expeditionParent[v]=u;expeditionDepth[v]=expeditionDepth[u]+1;
    expeditionLoss[v]=expeditionLoss[u]+int(!mine(v))+guard;heap.push({next,v});
   }
  }
  int goals[10],goalCount=0;double goalPriority[10];
  for(int c=0;c<N;c++)if(intelGain[c]>=2.5&&expeditionDepth[c]>0&&expeditionDepth[c]<=24){
   double value=intelGain[c]/(2.+.25*expeditionCost[c]);
   if(gd[c]<8)value*=.5;
   int j=goalCount;if(j==10){j=9;if(value<=goalPriority[j])continue;}else goalCount++;
   while(j>0&&value>goalPriority[j-1]){if(j<10){goals[j]=goals[j-1];goalPriority[j]=goalPriority[j-1];}j--;}
   goals[j]=c;goalPriority[j]=value;
  }
  Program best;
  for(int gi=0;gi<goalCount;gi++){
   int goal=goals[gi],path[25],n=0;
   for(int u=goal;u!=root&&u>=0&&n<25;u=expeditionParent[u])path[n++]=u;
   if(!n||n>24)continue;
   std::fill(expeditionVision,expeditionVision+N,0);
   double information=0;int threatOnPath=0;
   for(int j=0;j<n;j++){
    int u=path[j];if(zoneETA[u]<=2)threatOnPath=std::max(threatOnPath,zoneThreat[u]);
    for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){
     int x=cellX[u]+dx,y=cellY[u]+dy;if(x<0||x>=W||y<0||y>=H)continue;
     int c=y*W+x;if(!expeditionVision[c]){expeditionVision[c]=1;information+=intel[c];}
    }
   }
   for(int k=0;k<=10;k++){
    int assembled=schedule[root][k];if(assembled<1)continue;
    bool half=requiredHalf||assembled>=90||(st[root]==1&&assembled>=35);
    int moving=half?assembled/2:assembled-1;
    if(assembled-moving<reserve[root]||moving<18||moving>100)continue;
    int arrival=moving-expeditionLoss[goal];
    if(arrival<std::max(8,threatOnPath+3))continue;
    double value=(8.+std::min(32.,information)*1.1)/(3.+.40*(k+n)+.12*expeditionLoss[goal]);
    value=std::min(10.,value);
    if(value<=best.value)continue;
    Program q;q.kind=16;q.root=root;q.goal=goal;q.born=tick;q.value=value;
    if(k)extractSchedule(root,k,q);
    int u=root;for(int j=n-1;j>=0;j--){q.orders[q.count++]={u,path[j],j==n-1&&half};u=path[j];}
    if(q.count&&permitted(q.orders[0]))best=q;
   }
  }
  offer(best);
 }
}
