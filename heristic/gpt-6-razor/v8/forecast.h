// Bounded counterfactual rollouts for short operational programmes. Both
// orders are selected before either executes. The two opposing continuations
// are hypotheses generated from this seat's remembered board, never calls to
// an opponent controller or access to that controller's information.
struct ForecastPosition {int owner[MAXN],army[MAXN];unsigned char revealed[MAXN];};
ForecastPosition forecastInitial,forecastBoard;
int forecastEnemy=-1,forecastDistance[2][MAXN];
double forecastProjection[2][MAXN];
double forecastInformationPrice=0;

bool forecastFinished(const ForecastPosition &s)const {
 return s.owner[g]!=p||(enemyGeneral[forecastEnemy]>=0&&s.owner[enemyGeneral[forecastEnemy]]==p);
}
void forecastMove(ForecastPosition &s,const Order &x,int player)const {
 if(x.s<0||x.t<0||s.owner[x.s]!=player||forecastFinished(s))return;
 int moving=x.half?s.army[x.s]/2:std::max(0,s.army[x.s]-1);
 s.army[x.s]-=moving;
 if(s.owner[x.t]==player)s.army[x.t]+=moving;
 else if(moving>s.army[x.t]){s.army[x.t]=moving-s.army[x.t];s.owner[x.t]=player;}
 else s.army[x.t]-=moving;
}
double forecastAsset(const ForecastPosition &s,int c)const {
 int owner=s.owner[c];if(owner!=p&&owner!=forecastEnemy)return 0;
 if(st[c]!=1&&st[c]!=2)return 0;
 int incoming=0;
 for(int k=0;k<4;k++){int v=nb[c][k];if(v>=0&&s.owner[v]>=0&&s.owner[v]!=owner)incoming=std::max(incoming,s.army[v]-1);}
 double retention=std::min(1.,double(s.army[c]+2)/(incoming+2.));
 return (owner==p?1.:-1.)*(st[c]==2?180.:45.)*retention;
}
double forecastTile(const ForecastPosition &s,int c,int profile)const {
 int owner=s.owner[c];if(owner!=p&&owner!=forecastEnemy)return 0;
 int side=owner==p?0:1;
 double advance=forecastProjection[side][c]*(profile==1&&side==1?2.:1.);
 return (side==0?1.:-1.)*(s.army[c]+.9+advance*std::max(0,s.army[c]-1));
}
double forecastValue(const ForecastPosition &s,int profile)const {
 if(s.owner[g]!=p)return -100000.;
 if(enemyGeneral[forecastEnemy]>=0&&s.owner[enemyGeneral[forecastEnemy]]==p)return 100000.;
 double value=0;int largest[2]{};
 for(int c=0;c<N;c++){
  value+=forecastTile(s,c,profile)+forecastAsset(s,c);
  if(s.revealed[c]&&!forecastInitial.revealed[c])value+=forecastInformationPrice*intel[c];
  if(s.owner[c]==p)largest[0]=std::max(largest[0],s.army[c]);
  if(s.owner[c]==forecastEnemy)largest[1]=std::max(largest[1],s.army[c]);
 }
 return value+.12*(largest[0]-largest[1]);
}
Order forecastOrder(ForecastPosition &s,int player,int profile){
 int roots[8],count=0;double priority[8];
 for(int c=0;c<N;c++)if(s.owner[c]==player&&s.army[c]>1){
  double rank=s.army[c];
  int king=player==p?g:enemyGeneral[forecastEnemy];
  if(king>=0&&direction(c,king)>=0)rank+=20.;
  int j=count;if(j==8){j=7;if(rank<=priority[j])continue;}else count++;
  while(j>0&&rank>priority[j-1]){if(j<8){roots[j]=roots[j-1];priority[j]=priority[j-1];}j--;}
  roots[j]=c;priority[j]=rank;
 }
 Order best;double bestGain=0;
 for(int ri=0;ri<count;ri++){
  int u=roots[ri];
  for(int k=0;k<4;k++){
   int v=nb[u][k];if(v<0||blocked(v))continue;
   int affected[10],affectedCount=0;
   auto add=[&](int c){if(c<0)return;for(int j=0;j<affectedCount;j++)if(affected[j]==c)return;affected[affectedCount++]=c;};
   add(u);add(v);for(int j=0;j<4;j++){add(nb[u][j]);add(nb[v][j]);}
   double before=forecastTile(s,u,profile)+forecastTile(s,v,profile);
   for(int j=0;j<affectedCount;j++)before+=forecastAsset(s,affected[j]);
   int oldU=s.army[u],oldV=s.army[v],oldOwner=s.owner[v];
   int beforeLargest=std::max(s.army[u],s.owner[v]==player?s.army[v]:0);
   for(int half=0;half<2;half++){
    if(half&&oldU<4)continue;
    Order x{u,v,bool(half)};forecastMove(s,x,player);
    double after=forecastTile(s,u,profile)+forecastTile(s,v,profile);
    for(int j=0;j<affectedCount;j++)after+=forecastAsset(s,affected[j]);
    double gain=(player==p?1.:-1.)*(after-before);
    gain+=.12*(std::max(s.army[u],s.owner[v]==player?s.army[v]:0)-beforeLargest);
    if(s.owner[g]!=p)gain=player==p?-100000.:100000.;
    if(enemyGeneral[forecastEnemy]>=0&&s.owner[enemyGeneral[forecastEnemy]]==p)gain=player==p?100000.:-100000.;
    if(gain>bestGain){bestGain=gain;best=x;}
    s.army[u]=oldU;s.army[v]=oldV;s.owner[v]=oldOwner;
   }
  }
 }
 return best;
}

double forecastRun(const Program *program,int profile,int horizon,bool &held){
 forecastBoard=forecastInitial;
 for(int step=0;step<horizon&&!forecastFinished(forecastBoard);step++){
  if(step)for(int c=0;c<N;c++)if(forecastBoard.owner[c]>=0)forecastBoard.army[c]+=grows(c,actionTurn(step+1));
  Order theirs=forecastOrder(forecastBoard,forecastEnemy,profile),ours;
  if(program&&program->cursor+step<program->count){
   ours=program->orders[program->cursor+step];
   if(forecastBoard.owner[ours.s]!=p)ours=forecastOrder(forecastBoard,p,profile);
  }else ours=forecastOrder(forecastBoard,p,profile);
  if(forecastEnemy<p)forecastMove(forecastBoard,theirs,forecastEnemy);
  forecastMove(forecastBoard,ours,p);
  if(forecastEnemy>p)forecastMove(forecastBoard,theirs,forecastEnemy);
  for(int c=0;c<N;c++)if(forecastBoard.owner[c]==p&&forecastInitial.owner[c]!=p){
   for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){
    int x=cellX[c]+dx,y=cellY[c]+dy;if(x>=0&&x<W&&y>=0&&y<H)forecastBoard.revealed[y*W+x]=1;
   }
  }
 }
 held=!program||forecastBoard.owner[program->goal]==p;
 return forecastValue(forecastBoard,profile);
}

void forecastPrograms(){
 if(aliveCount!=2||tick<150||baselineDanger>1||proposalCount<2||!planningTimeLeft())return;
 forecastEnemy=-1;for(int e=0;e<o->players;e++)if(e!=p&&o->alive[e])forecastEnemy=e;
 double information=0;
 for(int c=0;c<N;c++){
  forecastInitial.owner[c]=own[c];forecastInitial.army[c]=mine(c)?ready(c):rememberedDefense(c);
  forecastInitial.revealed[c]=seen[c]>=0;information+=intel[c];
 }
 forecastInformationPrice=enemyGeneral[forecastEnemy]<0?.30*eliminationValue(forecastEnemy)/std::max(1.,information):0.;
 for(int side=0;side<2;side++){
  int head=0,tail=0;
  for(int c=0;c<N;c++){
   forecastDistance[side][c]=INF;
   if((side==0&&own[c]==forecastEnemy&&st[c]>0&&st[c]<3)||(side==1&&c==g)){
    forecastDistance[side][c]=0;queue[tail++]=c;
   }
  }
  if(tail==0)for(int c=0;c<N;c++)if(own[c]==forecastEnemy){forecastDistance[side][c]=0;queue[tail++]=c;}
  while(head<tail){int u=queue[head++];for(int k=0;k<4;k++){
   int v=nb[u][k];if(v<0||blocked(v)||forecastDistance[side][v]<INF)continue;
   forecastDistance[side][v]=forecastDistance[side][u]+1;queue[tail++]=v;
  }}
  for(int c=0;c<N;c++)forecastProjection[side][c]=forecastDistance[side][c]<INF?.30/(1.+.25*forecastDistance[side][c]):0.;
 }
 int indices[18];for(int j=0;j<proposalCount;j++)indices[j]=j;
 std::sort(indices,indices+proposalCount,[&](int a,int b){return proposalScore[a]>proposalScore[b];});
 double fallback[2];bool held;
 for(int profile=0;profile<2;profile++){
  if(!planningTimeLeft())return;
  fallback[profile]=forecastRun(nullptr,profile,8,held);
 }
 int evaluated=0;
 for(int j=0;j<proposalCount&&evaluated<6&&planningTimeLeft();j++){
  int index=indices[j];const auto &q=proposals[index];
  if(q.kind==3||q.kind==4||q.kind==9||q.kind==17||q.kind==21||q.goal<0||q.count-q.cursor>6)continue;
  const auto &x=q.orders[q.cursor];
  if(st[q.goal]!=1&&st[x.s]!=1&&st[x.t]!=1)continue;
  double worst=1e30;bool lostGoal=false,complete=true;
  for(int profile=0;profile<2;profile++){
   if(!planningTimeLeft()){complete=false;break;}
   double value=forecastRun(&q,profile,8,held)-fallback[profile];
   worst=std::min(worst,value);
   if(st[q.goal]==1&&!mine(q.goal)&&!held)lostGoal=true;
  }
  if(!complete)break;evaluated++;
  proposalScore[index]+=std::clamp(worst*.12,-6.,3.);
  if(lostGoal&&(q.kind==1||q.kind==23))proposalScore[index]-=.65*std::max(0.,q.value);
 }
}
