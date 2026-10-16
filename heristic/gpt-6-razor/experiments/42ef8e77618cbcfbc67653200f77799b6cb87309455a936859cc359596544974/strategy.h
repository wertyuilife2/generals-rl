// Objective-first operations: select a productive target, select its entrance,
// and price one concrete collection-and-occupation program. The terminal
// position includes the army left at the objective, not just the move there.
double objectiveCost[MAXN],objectiveStock[MAXN];
int objectiveNext[MAXN],objectiveSteps[MAXN],objectiveLoss[MAXN];

void occupationPrograms(){
 if(tick<150||baselineDanger>1)return;
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
   scheduleTree(root,budget);Program best;
   for(int k=0;k<=budget;k++)if(schedule[root][k]>0){
    Program q;q.kind=23;q.root=root;q.goal=goal;q.born=tick;
    if(k)extractSchedule(root,k,q);
    int assembled=k?collectionArrival(q,root):ready(root);
    bool half=reserve[root]>1;int mobile=half?assembled/2:assembled-1;
    if(assembled-mobile<reserve[root])continue;
    int steps=q.count+distance;
    int remaining=mobile-objectiveLoss[root]+1-futureGrowth(steps);
    if(remaining<8)continue;
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
