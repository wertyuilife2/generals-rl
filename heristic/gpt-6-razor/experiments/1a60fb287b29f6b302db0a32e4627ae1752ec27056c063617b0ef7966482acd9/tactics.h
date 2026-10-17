// Local, observation-derived two-round game trees. A joint round respects
// seat order; each side chooses its order before seeing the other's order.
struct TacticalPosition {int owner[41],army[41];};
struct TacticalOrder {int s=-1,t=-1;bool half=false;double rank=0;};
int tacticalCells[41],tacticalNeighbors[41][4],tacticalCount=0,tacticalEnemy=-1;
int tacticalHome=-1,tacticalKing=-1,tacticalWork=0;
int tacticalFocus=-1;
int tacticalProgress[2][41];
double tacticalProjection[2][41];
bool tacticalAborted=false;

double tacticalTile(int i,int owner,int army)const {
 if(owner!=p&&owner!=tacticalEnemy)return 0;
 int c=tacticalCells[i];double value=army+.8+(st[c]==1?28.:0.);
 value+=tacticalProjection[owner==p?0:1][i]*std::max(0,army-1);
 return owner==p?value:-value;
}

double tacticalValue(const TacticalPosition &s)const {
 if(tacticalHome>=0&&s.owner[tacticalHome]!=p)return -100000.;
 if(tacticalKing>=0&&s.owner[tacticalKing]==p)return 100000.;
 double result=0;int friendly=0,hostile=0;
 for(int i=0;i<tacticalCount;i++){
  result+=tacticalTile(i,s.owner[i],s.army[i]);
  if(s.owner[i]==p)friendly=std::max(friendly,s.army[i]-1);
  if(s.owner[i]==tacticalEnemy)hostile=std::max(hostile,s.army[i]-1);
 }
 return result+30.*(friendly-hostile)/(friendly+hostile+20.);
}

void tacticalMove(TacticalPosition &s,const TacticalOrder &order,int player)const {
 if(order.s<0||s.owner[order.s]!=player||
    (tacticalHome>=0&&s.owner[tacticalHome]!=p)||
    (tacticalKing>=0&&s.owner[tacticalKing]==p))return;
 int sent=order.half?s.army[order.s]/2:std::max(0,s.army[order.s]-1);
 s.army[order.s]-=sent;
 if(s.owner[order.t]==player)s.army[order.t]+=sent;
 else if(sent>s.army[order.t]){s.army[order.t]=sent-s.army[order.t];s.owner[order.t]=player;}
 else s.army[order.t]-=sent;
}

void tacticalRound(TacticalPosition &s,const TacticalOrder &us,const TacticalOrder &them)const {
 if(tacticalEnemy<p)tacticalMove(s,them,tacticalEnemy);
 tacticalMove(s,us,p);
 if(tacticalEnemy>p)tacticalMove(s,them,tacticalEnemy);
}

int tacticalOrders(const TacticalPosition &s,int player,TacticalOrder *orders,int limit){
 int count=1;orders[0]={-1,-1,false,0};
 for(int u=0;u<tacticalCount;u++)if(s.owner[u]==player&&s.army[u]>1){
  for(int k=0;k<4;k++){
   int v=tacticalNeighbors[u][k];if(v<0)continue;
   if(player==p&&std::abs(cellX[tacticalCells[v]]-cellX[tacticalFocus])+
      std::abs(cellY[tacticalCells[v]]-cellY[tacticalFocus])>3)continue;
   for(int half=0;half<2;half++){
    if(half&&s.army[u]<4)continue;
    if(++tacticalWork>24000||((tacticalWork&255)==0&&!planningTimeLeft())){tacticalAborted=true;return count;}
    int sent=half?s.army[u]/2:s.army[u]-1,owner=s.owner[v],army=s.army[v];
    if(owner==player)army+=sent;
    else if(sent>army){army=sent-army;owner=player;}
    else army-=sent;
    double gain=tacticalTile(u,player,s.army[u]-sent)+tacticalTile(v,owner,army)
               -tacticalTile(u,player,s.army[u])-tacticalTile(v,s.owner[v],s.army[v]);
    if(player!=p)gain=-gain;
    if(s.owner[v]>=0&&s.owner[v]!=player)gain+=.04*std::min(sent,s.army[v]);
    if(s.owner[v]==player)gain+=.005*sent;
    if(v==tacticalHome||v==tacticalKing)gain+=sent>s.army[v]?10000.:0.;
    int j=count;if(j==limit){j=limit-1;if(gain<=orders[j].rank)continue;}else count++;
    // Keep WAIT as an explicit response, independently of move ordering.
    while(j>1&&gain>orders[j-1].rank){if(j<limit)orders[j]=orders[j-1];j--;}
    orders[j]={u,v,bool(half),gain};
   }
  }
 }
 return count;
}

double tacticalSecondRound(const TacticalPosition &s,double beta){
 TacticalOrder ours[6],theirs[6];int nc=tacticalOrders(s,p,ours,6),ne=tacticalOrders(s,tacticalEnemy,theirs,6);
 double best=-1e30;
 for(int i=0;i<nc&&!tacticalAborted;i++){
  double worst=1e30;
  for(int j=0;j<ne;j++){
   TacticalPosition next=s;tacticalRound(next,ours[i],theirs[j]);
   worst=std::min(worst,tacticalValue(next));if(worst<=best)break;
  }
  best=std::max(best,worst);if(best>=beta)break;
 }
 return best;
}

double tacticalFirstOrder(const TacticalPosition &s,const TacticalOrder &us,const TacticalOrder *replies,int count,double alpha){
 double worst=1e30;
 for(int j=0;j<count&&!tacticalAborted;j++){
  TacticalPosition next=s;tacticalRound(next,us,replies[j]);
  for(int c=0;c<tacticalCount;c++)if(next.owner[c]>=0)next.army[c]+=grows(tacticalCells[c],actionTurn(2));
  worst=std::min(worst,tacticalSecondRound(next,worst));
  if(worst<=alpha)break;
 }
 return worst;
}

void localBattlePrograms(){
 if(aliveCount!=2||tick<120||baselineDanger>0)return;
 tacticalEnemy=-1;for(int e=0;e<o->players;e++)if(e!=p&&o->alive[e])tacticalEnemy=e;
 int focus=-1;double priority=-1;
 // A mobile army can be lost on ordinary ground while unrelated economy
 // keeps issuing orders. Give that contact the same joint reply search as
 // a city fight; the tree still compares escape, engagement and waiting.
 for(int c=0;c<N;c++)if(o->visible[c]&&own[c]>=0&&
    (st[c]==1||(mine(c)&&st[c]==0&&ready(c)>=12))){
  int us=mine(c)?ready(c):0,them=mine(c)?0:ready(c);
  for(int dy=-3;dy<=3;dy++)for(int dx=-3;dx<=3;dx++){
   if(std::abs(dx)+std::abs(dy)>3)continue;
   int x=cellX[c]+dx,y=cellY[c]+dy;if(x<0||x>=W||y<0||y>=H)continue;
   int v=y*W+x;if(!o->visible[v])continue;
   if(mine(v))us=std::max(us,ready(v));else if(own[v]==tacticalEnemy)them=std::max(them,ready(v));
  }
  if(us<8||them<8)continue;
  double value=(st[c]==1?80.:0)+std::min(200,std::min(us,them));
  if(value>priority){priority=value;focus=c;}
 }
 if(focus<0)return;tacticalFocus=focus;
 tacticalCount=0;tacticalHome=tacticalKing=-1;tacticalWork=0;tacticalAborted=false;
 for(int c=0;c<N;c++)if(!blocked(c)&&
    std::abs(cellX[c]-cellX[focus])+std::abs(cellY[c]-cellY[focus])<=4)tacticalCells[tacticalCount++]=c;
 TacticalPosition initial;
 for(int i=0;i<tacticalCount;i++){
  int c=tacticalCells[i];initial.owner[i]=own[c];initial.army[i]=mine(c)?ready(c):rememberedDefense(c);
  if(c==g)tacticalHome=i;
  if(c==enemyGeneral[tacticalEnemy])tacticalKing=i;
  for(int k=0;k<4;k++){
   tacticalNeighbors[i][k]=-1;
   for(int j=0;j<tacticalCount;j++)if(tacticalCells[j]==nb[c][k])tacticalNeighbors[i][k]=j;
  }
 }
 for(int side=0;side<2;side++){
  int targetOwner=side==0?tacticalEnemy:p,work[41],head=0,tail=0;
  for(int i=0;i<tacticalCount;i++){
   tacticalProgress[side][i]=INF;
   if(initial.owner[i]==targetOwner&&(initial.army[i]>=8||st[tacticalCells[i]]>0)){
    tacticalProgress[side][i]=0;work[tail++]=i;
   }
  }
  while(head<tail){int u=work[head++];
   for(int k=0;k<4;k++){int v=tacticalNeighbors[u][k];if(v<0||tacticalProgress[side][v]<INF)continue;
    tacticalProgress[side][v]=tacticalProgress[side][u]+1;work[tail++]=v;
   }
  }
  for(int i=0;i<tacticalCount;i++)tacticalProjection[side][i]=
   tacticalProgress[side][i]<INF?.15/(1.+.25*tacticalProgress[side][i]):0.;
 }
 TacticalOrder ours[13],replies[6];int count=tacticalOrders(initial,p,ours,9),nr=tacticalOrders(initial,tacticalEnemy,replies,6);
 // Include the scheduler's strongest local proposals even when they are
 // quiet transfers omitted by the immediate-gain move ordering.
 int indices[18];for(int j=0;j<proposalCount;j++)indices[j]=j;
 std::sort(indices,indices+proposalCount,[&](int a,int b){return proposalScore[a]>proposalScore[b];});
 for(int j=0;j<proposalCount&&count<13;j++){
  auto &plan=proposals[indices[j]];auto x=plan.orders[plan.cursor];int s=-1,t=-1;
  for(int c=0;c<tacticalCount;c++){if(tacticalCells[c]==x.s)s=c;if(tacticalCells[c]==x.t)t=c;}
  if(s<0||t<0)continue;
  bool exists=false;for(int c=0;c<count;c++)if(ours[c].s==s&&ours[c].t==t&&ours[c].half==x.half)exists=true;
  if(!exists)ours[count++]={s,t,x.half,0};
 }
 double wait=tacticalFirstOrder(initial,ours[0],replies,nr,-1e30),best=wait;
 if(tacticalAborted)return;
 int selected=-1;
 for(int i=1;i<count&&!tacticalAborted;i++){
  Order action{tacticalCells[ours[i].s],tacticalCells[ours[i].t],ours[i].half};
  if(!legalOrder(action,true))continue;
  if(landingCanBeDenied(action))continue;
  if(std::abs(cellX[action.s]-cellX[focus])+std::abs(cellY[action.s]-cellY[focus])>2||
     std::abs(cellX[action.t]-cellX[focus])+std::abs(cellY[action.t]-cellY[focus])>2)continue;
  if(dangerAfter(&action)>baselineDanger+1)continue;
  double value=tacticalFirstOrder(initial,ours[i],replies,nr,best);
  if(!tacticalAborted&&value<wait-2.){
   // Alpha pruning returns an upper bound for a rejected root action.
   // Even that optimistic bound losing to WAIT is sufficient to reduce
   // the scheduler's price for the same move. Quiet actions elsewhere
   // remain available; this does not force a global WAIT.
   for(int j=0;j<proposalCount;j++){
    const auto &q=proposals[j];const auto &x=q.orders[q.cursor];
    if(q.kind==3||q.kind==4||q.kind==9||q.kind==17)continue;
    if(x.s==action.s&&x.t==action.t&&x.half==action.half)
     proposalScore[j]-=std::min(40.,.4*(wait-value));
   }
  }
  if(!tacticalAborted&&value>best+.01){best=value;selected=i;}
 }
 if(selected<0||best-wait<3.)return;
 auto chosen=ours[selected];int s=tacticalCells[chosen.s],t=tacticalCells[chosen.t];
 offer(oneStep({std::min(14.,3.+.5*(best-wait)),s,t,direction(s,t),chosen.half},20));
}
