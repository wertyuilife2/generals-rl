// A proposed order spends one command and real troops. Rebuild a concrete
// response from its resulting position and test it on a shared troop ledger.
// This is a bounded single-column model, not an exhaustive safety proof.
struct ResponseRoute {int cells[25]{},count=0;};
ResponseRoute responseRoutes[4];
int responseCount=0,responseEnemy=-1,responseHorizon=0;
bool responseActive=false,responseLongRecall=false;
Program responseBaseline;
int responseIndex[MAXN],responseCells[256],responseParent[256],responseDepth[256];
int responseChildren[256][4],responseForce[256][25];
signed char responsePick[256][4][25];

bool responseOutcome(const Program &plan,const ResponseRoute &route)const {
 int index[MAXN],cells[128],owners[128],troops[128],count=0;
 std::fill(index,index+N,-1);
 auto add=[&](int c){if(index[c]>=0)return index[c];
  if(count==128)return -1;
  int i=count++;index[c]=i;cells[i]=c;owners[i]=own[c];
  troops[i]=mine(c)?ready(c):rememberedDefense(c);return i;};
 add(g);for(int j=0;j<=route.count;j++)if(add(route.cells[j])<0)return false;
 for(int j=0;j<plan.count;j++)if(plan.orders[j].s>=0)
  for(int c:{plan.orders[j].s,plan.orders[j].t})if(add(c)<0)return false;
 int enemy=own[responseEnemy],position=0;bool advancing=true;
 auto enemyMove=[&](){
  if(!advancing||position>=route.count)return;
  int s=index[route.cells[position]],t=index[route.cells[position+1]];
  if(owners[s]!=enemy||troops[s]<=1){advancing=false;return;}
  int sent=troops[s]-1;troops[s]=1;
  if(owners[t]==enemy){troops[t]+=sent;position++;}
  else if(sent>troops[t]){troops[t]=sent-troops[t];owners[t]=enemy;position++;}
  else{troops[t]-=sent;advancing=false;}
 };
 auto ourMove=[&](int step){
  if(step>=plan.count||plan.orders[step].s<0)return true;
  auto x=plan.orders[step];int s=index[x.s],t=index[x.t];
  if(owners[s]!=p)return false;
  int sent=x.half?troops[s]/2:troops[s]-1;
  if(sent<=0)return false;
  if(step>0&&troops[s]-sent<reserve[x.s])return false;
  troops[s]-=sent;
  if(owners[t]==p)troops[t]+=sent;
  else if(sent>troops[t]){troops[t]=sent-troops[t];owners[t]=p;}
  else{troops[t]-=sent;return false;}
  return true;
 };
 for(int step=0;step<route.count;step++){
  if(step)for(int i=0;i<count;i++)if(owners[i]>=0)
   troops[i]+=grows(cells[i],actionTurn(step+1));
  if(enemy<p)enemyMove();
  if(owners[index[g]]!=p)return false;
  if(!ourMove(step))return false;
  if(st[responseEnemy]==2&&owners[index[responseEnemy]]==p)return true;
  if(enemy>p)enemyMove();
  if(owners[index[g]]!=p)return false;
 }
 return true;
}

void responseExtract(int u,int budget,Program &q){
 int use[4];
 for(int k=3;k>=0;k--){use[k]=responsePick[u][k][budget];if(use[k]>=0)budget-=use[k]+1;}
 for(int k=0;k<4;k++)if(use[k]>=0){
  int child=responseChildren[u][k];responseExtract(child,use[k],q);
  q.orders[q.count++]={responseCells[child],responseCells[u],reserve[responseCells[child]]>1};
 }
}

bool responsePlan(const Order *prefix,Program &out){
 int sent=prefix&&prefix->s>=0?moveArmy(prefix->s,prefix->half):0;
 auto ownerAfter=[&](int c){
  if(prefix&&c==prefix->t&&!mine(c)&&sent>rememberedDefense(c))return p;
  return own[c];
 };
 auto forceAfter=[&](int c){
  int n=mine(c)?ready(c):rememberedDefense(c);
  if(prefix&&prefix->s>=0){
   if(c==prefix->s)n-=sent;
   if(c==prefix->t)n=mine(c)?n+sent:sent-n;
  }
  return n;
 };
 int budget=std::min(24,responseHorizon-int(prefix!=nullptr));
 if(budget<0)return false;
 std::fill(responseIndex,responseIndex+N,-1);
 int count=1;responseCells[0]=g;responseParent[0]=-1;responseDepth[0]=0;responseIndex[g]=0;
 for(int i=0;i<count;i++){
  int c=responseCells[i];
  for(int k=0;k<4;k++){
   responseChildren[i][k]=-1;int v=nb[c][k];
   if(v<0||ownerAfter(v)!=p||responseIndex[v]>=0||responseDepth[i]>=budget||count==256)continue;
   int j=count++;responseCells[j]=v;responseIndex[v]=j;responseParent[j]=i;
   responseDepth[j]=responseDepth[i]+1;responseChildren[i][k]=j;
  }
 }
 for(int u=count-1;u>=0;u--){
  responseForce[u][0]=forceAfter(responseCells[u]);
  for(int b=1;b<=budget;b++)responseForce[u][b]=-INF;
  for(int k=0;k<4;k++){
   for(int b=0;b<=budget;b++)responsePick[u][k][b]=-1;
   int child=responseChildren[u][k];if(child<0)continue;
   int guard=reserve[responseCells[child]];
   for(int b=budget;b>=1;b--)for(int used=0;used<b;used++){
    int previous=responseForce[u][b-used-1],mass=responseForce[child][used];
    if(previous<0||mass<=1)continue;
    int available=guard>1?((mass+1)/2>=guard?mass/2:0):mass-1;
    if(available>0&&previous+available>responseForce[u][b]){
     responseForce[u][b]=previous+available;responsePick[u][k][b]=used;
    }
   }
  }
 }
 // More collection is not assumed automatically safer: an earlier raid
 // may cut a donor or the corridor while the sequence is in progress.
 for(int used=0;used<=budget;used++)if(responseForce[0][used]>=0){
  Program q;q.kind=3;q.root=q.goal=g;q.born=tick;q.value=24;
  if(prefix)q.orders[q.count++]=*prefix;
  responseExtract(0,used,q);
  bool safe=true;for(int j=0;j<responseCount&&safe;j++)safe=responseOutcome(q,responseRoutes[j]);
  if(safe){out=q;return true;}
 }
 return false;
}

void prepareResponseBudget(){
 responseActive=false;responseLongRecall=false;responseEnemy=-1;responseCount=0;
 if(aliveCount!=2||tick<120||baselineDanger>0)return;
 int second=0;
 for(int c=0;c<N;c++)if(o->visible[c]&&own[c]>=0&&own[c]!=p&&gd[c]<=24&&ready(c)>=20){
  if(responseEnemy<0||ready(c)>ready(responseEnemy)){
   if(responseEnemy>=0)second=std::max(second,ready(responseEnemy));responseEnemy=c;
  }else second=std::max(second,ready(c));
 }
 if(responseEnemy<0||gd[responseEnemy]<=8||ready(responseEnemy)<second*3)return;
 if(mobilizationETA[responseEnemy]>=INF||mobilizationThreat[responseEnemy]<=army(g)+1)return;
 responseHorizon=gd[responseEnemy];
 for(int rotation=0;rotation<4;rotation++){
  ResponseRoute r;r.cells[0]=responseEnemy;r.count=0;int u=responseEnemy;
  while(u!=g&&r.count<24){
   int v=-1;for(int k=0;k<4;k++){
    int candidate=nb[u][(k+rotation)%4];
    if(candidate>=0&&!blocked(candidate)&&gd[candidate]+1==gd[u]){v=candidate;break;}
   }
   if(v<0)break;r.cells[++r.count]=v;u=v;
  }
  if(u!=g)continue;
  bool duplicate=false;for(int j=0;j<responseCount;j++)
   duplicate|=r.count==responseRoutes[j].count&&std::equal(r.cells,r.cells+r.count+1,responseRoutes[j].cells);
  if(!duplicate)responseRoutes[responseCount++]=r;
 }
 if(responseCount)responseActive=responsePlan(nullptr,responseBaseline);
}

bool selectResponseBudget(Program &chosen){
 if(!responseActive||!responseLongRecall)return false;
 int order[18];for(int j=0;j<proposalCount;j++)order[j]=j;
 std::sort(order,order+proposalCount,[&](int a,int b){return proposalScore[a]>proposalScore[b];});
 int attempts=0;
 for(int z=0;z<proposalCount&&attempts<4;z++){
  int j=order[z];const Program &q=proposals[j];
  if(q.kind==3||proposalScore[j]<=-.01-baselineDanger)continue;
  auto x=q.orders[q.cursor];
  if(q.kind==4&&st[x.t]==2&&own[x.t]>=0&&own[x.t]!=p){chosen=q;return true;}
  attempts++;Program response;
  if(responsePlan(&x,response)){chosen=q;return true;}
 }
 chosen=responseBaseline;return true;
}
