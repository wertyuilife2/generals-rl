// Objective-first operations: select a productive target, select its entrance,
// and price one concrete collection-and-occupation program. The terminal
// position includes the army left at the objective, not just the move there.
double objectiveCost[MAXN],objectiveStock[MAXN];
int objectiveNext[MAXN],objectiveSteps[MAXN],objectiveLoss[MAXN];
int occupationRoots[8],occupationForce[8][21],occupationRootCount=0;
Program occupationCollections[8][21];

int occupationCollection(int root){
 for(int i=0;i<occupationRootCount;i++)if(occupationRoots[i]==root)return i;
 if(occupationRootCount==8)return -1;
 int index=occupationRootCount++;occupationRoots[index]=root;scheduleTree(root,20);
 for(int k=0;k<=20;k++){
  auto &q=occupationCollections[index][k];q=Program();q.kind=23;q.root=root;q.born=tick;
  occupationForce[index][k]=-INF;if(schedule[root][k]<1)continue;
  if(k)extractSchedule(root,k,q);
  occupationForce[index][k]=k?collectionArrival(q,root):ready(root);
 }
 return index;
}

void occupationPrograms(){
 if(tick<150||baselineDanger>1)return;
 occupationRootCount=0;
 int goals[6],goalCount=0;double priority[6];
 for(int c=0;c<N;c++)if(own[c]>=0&&own[c]!=p&&o->alive[own[c]]&&(st[c]==1||st[c]==2)){
  int distance=INF;
  for(int u:sources)distance=std::min(distance,std::abs(cellX[u]-cellX[c])+std::abs(cellY[u]-cellY[c]));
  double value=(st[c]==2?3.:1.)/(3.+distance);
  int j=goalCount;if(j==6){j=5;if(value<=priority[j])continue;}else goalCount++;
  while(j>0&&value>priority[j-1]){if(j<6){goals[j]=goals[j-1];priority[j]=priority[j-1];}j--;}
  goals[j]=c;priority[j]=value;
 }
 if(!goalCount)return;
 for(int c=0;c<N;c++){
  objectiveStock[c]=0;if(!mine(c))continue;
  bool half=false;if(canDepart(c,half))objectiveStock[c]=moveArmy(c,half);
  // This estimate orders candidates only. The exact disjoint tree below
  // determines the force that is actually available and the orders it costs.
  for(int k=0;k<4;k++){
   int v=nb[c][k];if(v<0||!mine(v))continue;
   if(canDepart(v,half))objectiveStock[c]+=.7*moveArmy(v,half);
   for(int z=0;z<4;z++){int u=nb[v][z];if(u>=0&&u!=c&&mine(u)&&canDepart(u,half))objectiveStock[c]+=.2*moveArmy(u,half);}
  }
 }
 double income=.5*(std::max<int64_t>(0,o->total_cities[p])+1)+.02*o->total_land[p];
 for(int gi=0;gi<goalCount;gi++){
  if(!planningTimeLeft())break;
  int goal=goals[gi],enemy=own[goal];bool king=st[goal]==2;
  std::fill(objectiveCost,objectiveCost+N,1e30);
  std::fill(objectiveSteps,objectiveSteps+N,INF);
  std::fill(objectiveNext,objectiveNext+N,-1);
  using Entry=std::pair<double,int>;auto &heap=campaignHeap;heap.clear();
  auto push=[&](Entry entry){heap.push_back(entry);std::push_heap(heap.begin(),heap.end(),std::greater<Entry>());};
  objectiveCost[goal]=0;objectiveSteps[goal]=0;objectiveLoss[goal]=0;push({0,goal});
  while(!heap.empty()){
   std::pop_heap(heap.begin(),heap.end(),std::greater<Entry>());auto [cost,u]=heap.back();heap.pop_back();
   if(cost>objectiveCost[u]+1e-8||objectiveSteps[u]>=20)continue;
   int guard=mine(u)?0:rememberedDefense(u)+int(ter[u]==255);
   for(int k=0;k<4;k++){
    int v=nb[u][k];if(v<0||blocked(v)||v==goal)continue;
    // A protected city may be a staging root, but not an imaginary full
    // transfer in the middle of the ensuing attack route.
    if(mine(u)&&reserve[u]>1)continue;
    double next=cost+1.+guard/(income*2.+5.);
    if(next>=objectiveCost[v]-1e-8)continue;
    objectiveCost[v]=next;objectiveSteps[v]=objectiveSteps[u]+1;
    objectiveLoss[v]=objectiveLoss[u]+1+guard;objectiveNext[v]=u;push({next,v});
   }
  }
  int roots[2]={-1,-1};double scores[2]={-1,-1};
  for(int c=0;c<N;c++)if(mine(c)&&objectiveSteps[c]>0&&objectiveSteps[c]<=18&&objectiveStock[c]>=8){
   if(c==g&&reserve[g]>1)continue;
   double score=std::sqrt(objectiveStock[c])/(2.+objectiveCost[c]);
   int j=score>scores[0]?0:score>scores[1]?1:-1;if(j<0)continue;
   if(j==0){roots[1]=roots[0];scores[1]=scores[0];}roots[j]=c;scores[j]=score;
  }
  for(int ri=0;ri<2;ri++)if(roots[ri]>=0){
   int root=roots[ri],distance=objectiveSteps[root],budget=std::min(20,38-distance);
   int collection=occupationCollection(root);if(collection<0)continue;
   Program best;
   for(int k=0;k<=budget;k++)if(occupationForce[collection][k]>0){
    Program q=occupationCollections[collection][k];q.goal=goal;
    int assembled=occupationForce[collection][k];
    bool half=reserve[root]>1;int mobile=half?assembled/2:assembled-1;
    if(assembled-mobile<reserve[root])continue;
    int steps=q.count+distance;
    int remaining=mobile-objectiveLoss[root]+1-futureGrowth(steps);
    if(remaining<8)continue;
    if(!king&&assetFlips[goal]>=2&&tick-assetLostAt[goal]<40&&remaining<zoneThreat[goal]+8)continue;
    double prize=0,confidence=1.;
    if(king){
     int relief=kingRelief[enemy][std::min(32,steps)];
     if(remaining<=relief+4)continue;
     prize=eliminationValue(enemy);
    }else{
     double response=(2.+cityDefenseHits[enemy])/(8.+cityDefenseCases[enemy]);
     int relief=assetRelief[goal][std::min(12,steps)];
     double expected=std::max(response*relief,.5*zoneThreat[goal]);
     confidence=std::clamp(remaining/(10.+expected),.1,1.);
     double horizon=120./(1.+.25*assetFlips[goal]);
     prize=12.+.5*horizon*(1.+1./std::max(1,aliveCount-1));
    }
    // Once the army exists, the next attempt is cheaper; do not price a
    // spent donor twice. Actual combat casualties still consume resources.
    q.value=confidence*prize/(steps+2.+objectiveLoss[root]/(income*3.+4.));
    int u=root;
    while(u!=goal&&q.count<40){int v=objectiveNext[u];if(v<0)break;
     q.orders[q.count++]={u,v,u==root&&half};u=v;
    }
    if(u==goal&&q.count&&q.value>best.value&&permitted(q.orders[0]))best=q;
   }
   offer(best);
  }
 }
}

int scoutingDistance[MAXN];
unsigned char scoutingRevealed[MAXN];
void informationPrograms(){
 if(tick<180||o->total_land[p]<45||baselineDanger>1||expeditionCredit<3)return;
 bool unknownKing=false;double totalInformation=0;
 for(int e=0;e<o->players;e++)if(e!=p&&o->alive[e]){
  if(enemyGeneral[e]<0)unknownKing=true;
 }
 if(!unknownKing)return;
 for(int c=0;c<N;c++)totalInformation+=kingInformation[c];
 if(totalInformation<2)return;
 int head=0,tail=0;
 for(int c=0;c<N;c++){
  scoutingDistance[c]=INF;
  if(!blocked(c)&&intelGain[c]>=2&&gd[c]>=8){scoutingDistance[c]=0;queue[tail++]=c;}
 }
 while(head<tail){int u=queue[head++];
  for(int k=0;k<4;k++){int v=nb[u][k];if(v<0||blocked(v)||scoutingDistance[v]<INF)continue;
   scoutingDistance[v]=scoutingDistance[u]+1;queue[tail++]=v;
  }
 }
 int roots[4],count=0;double priorities[4];
 for(int c=0;c<N;c++)if(mine(c)&&scoutingDistance[c]<=10){
  bool half=false;if(!canDepart(c,half))continue;
  if(ready(c)>36||st[c]>0)half=true;
  int mobile=moveArmy(c,half);if(mobile<8)continue;
  double value=std::min(1.,mobile/20.)/(1.+scoutingDistance[c]+.015*mobile);
  int j=count;if(j==4){j=3;if(value<=priorities[j])continue;}else count++;
  while(j>0&&value>priorities[j-1]){if(j<4){roots[j]=roots[j-1];priorities[j]=priorities[j-1];}j--;}
  roots[j]=c;priorities[j]=value;
 }
 for(int ri=0;ri<count;ri++){
  if(!planningTimeLeft())break;
  int source=roots[ri];bool half=false;canDepart(source,half);
  if(ready(source)>36||st[source]>0)half=true;
  int mobile=moveArmy(source,half);taskRoutes(source,mobile,true);
  int goals[8],goalCount=0;double ranks[8];
  for(int c=0;c<N;c++)if(!mine(c)&&opSteps[c]>0&&opSteps[c]<=std::min(16,int(expeditionCredit))&&opForce[c]>=6&&intelGain[c]>=2){
   double rank=intelGain[c]/(1.+routeCost[c]);int j=goalCount;
   if(j==8){j=7;if(rank<=ranks[j])continue;}else goalCount++;
   while(j>0&&rank>ranks[j-1]){if(j<8){goals[j]=goals[j-1];ranks[j]=ranks[j-1];}j--;}
   goals[j]=c;ranks[j]=rank;
  }
  Program best;
  for(int gi=0;gi<goalCount;gi++){
   int goal=goals[gi],path[17],n=0;
   for(int c=goal;c!=source&&c>=0&&n<17;c=opParent[c])path[n++]=c;
   if(n<1||n>16)continue;
   std::fill(scoutingRevealed,scoutingRevealed+N,0);
   double information=0,land=0;int spent=0;
   for(int j=n-1;j>=0;j--){
    int u=path[j];if(!mine(u)){
     land+=plainUtility(u,opForce[u],n-j);spent+=rememberedDefense(u)+1;
    }
    for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){
     int x=cellX[u]+dx,y=cellY[u]+dy;if(x<0||x>=W||y<0||y>=H)continue;
     int c=y*W+x;if(!scoutingRevealed[c]){scoutingRevealed[c]=1;information+=kingInformation[c];}
    }
   }
   double intelValue=informationPrice*information;
   double value=(land+intelValue)/(n+1.+.08*spent+.01*mobile);
   Program q;q.kind=24;q.root=source;q.goal=goal;q.born=tick;q.value=value;
   int u=source;for(int j=n-1;j>=0;j--){q.orders[q.count++]={u,path[j],j==n-1?half:opHalf[path[j]]};u=path[j];}
   if(q.value>best.value&&permitted(q.orders[0]))best=q;
  }
  offer(best);
 }
}

// Build a counter-column before an observed field army reaches the front.
// A programme must reach the required strength using disjoint donors within
// the enemy's travel deadline. Moving an already sufficient army is not a
// concentration benefit and cannot trigger this operation repeatedly.
void frontFormationPrograms(){
 if(tick<120||baselineDanger>1)return;
 int enemies[3],count=0;
 for(int c=0;c<N;c++)if(o->visible[c]&&own[c]>=0&&own[c]!=p&&st[c]==0&&army(c)>=20){
  int j=count;if(j==3){j=2;if(army(c)<=army(enemies[j]))continue;}else count++;
  while(j>0&&army(c)>army(enemies[j-1])){if(j<3)enemies[j]=enemies[j-1];j--;}
  enemies[j]=c;
 }
 for(int ei=0;ei<count;ei++){
  if(!planningTimeLeft())break;
  int enemy=enemies[ei],need=ready(enemy)+10,donor=0;
  for(int k=0;k<4;k++){int v=nb[enemy][k];if(v>=0&&o->visible[v]&&own[v]==own[enemy])donor=std::max(donor,moveArmy(v));}
  need+=donor/3;
  if(need>.7*std::max<int64_t>(0,o->total_army[p]-o->total_land[p]))continue;
  bfs(enemy,opSteps,false,true);
  int roots[3],rootCount=0,deadlines[3];double scores[3];
  // A one-order merge is still possible with an attacker one or two steps
  // away. Validate that joint round below instead of excluding the root
  // merely because its unreinforced garrison currently loses the fight.
  for(int c=0;c<N;c++)if(mine(c)&&c!=g&&opSteps[c]>=1&&opSteps[c]<=8){
   int potential=ready(c);
   for(int k=0;k<4;k++){int v=nb[c][k];if(v>=0&&mine(v)){
    bool half=false;if(canDepart(v,half))potential+=moveArmy(v,half);
   }}
   if(potential<need*.65)continue;
   double score=std::min(need,potential)/(2.+.3*opSteps[c]);
   int j=rootCount;if(j==3){j=2;if(score<=scores[j])continue;}else rootCount++;
   while(j>0&&score>scores[j-1]){if(j<3){roots[j]=roots[j-1];scores[j]=scores[j-1];deadlines[j]=deadlines[j-1];}j--;}
   roots[j]=c;scores[j]=score;deadlines[j]=std::clamp(opSteps[c]-1,1,6);
  }
  for(int ri=0;ri<rootCount;ri++){
   int root=roots[ri],budget=deadlines[ri];scheduleTree(root,budget);Program best;
   for(int k=1;k<=budget;k++)if(schedule[root][k]>0){
    Program q;q.kind=25;q.root=q.goal=root;q.born=tick;extractSchedule(root,k,q);
    int assembled=collectionArrival(q,root);
    int mobile=reserve[root]>1?assembled/2:assembled-1;
    if(assembled-mobile<reserve[root]||mobile<need)continue;
    int useful=mobileAssemblyCredit(q,root,mobile+1,need+1);
    if(useful<8)continue;
    q.value=std::min(10.,.65*useful/(k+2.)+12.*useful/(need+4.)/(1.+.3*k));
    if(q.value>best.value&&q.count&&permitted(q.orders[0]))best=q;
   }
   offer(best);
  }
 }
}
