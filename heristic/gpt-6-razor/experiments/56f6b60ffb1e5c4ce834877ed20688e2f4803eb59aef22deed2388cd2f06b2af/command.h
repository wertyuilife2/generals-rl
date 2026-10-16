// A field operation owns its army until it reaches an information objective,
// is redirected by a military emergency, or loses the ability to continue.
// The objective survives individual moves; a neighbouring cheap tile is not
// a new operation. Every remembered location comes from this seat's receipts.
int commandColumn=-1,commandGoal=-1,commandBorn=-1000,commandProgress=-1000;
int commandMinimum=0,commandNeed=0;
int commandFrom=-1,commandTo=-1,commandIssued=-1000,commandExpected=0;
int commandBefore=0,commandMoving=0;
bool commandHalf=false,commandActive=false;
bool commandReturn=false,commandTravels=false;
short commandHomeDistance[MAXN];
std::int64_t commandAction=0;
unsigned char commandReserved[MAXN]{};
Order commandNext;

bool columnClaimed(int c)const {
 for(const auto &task:operationTasks){
  const auto &q=task.plan;
  if(!q.kind||q.cursor>=q.count||tick-q.born>64||tick-task.progress>20)continue;
  if(q.kind==1&&(q.goal<0||st[q.goal]==0||st[q.goal]==255))continue;
  if(q.kind==23&&mine(q.goal))continue;
  if(task.pending&&o->previous&&o->previous->action==task.action&&!o->previous->valid)continue;
  if(!mine(q.orders[q.cursor].s)||blocked(q.orders[q.cursor].t))continue;
  if(q.root==c||(task.pending&&task.destination==c))return true;
  for(int j=q.cursor;j<q.count;j++)if(q.orders[j].s==c)return true;
 }
 return false;
}

void observeCommand(){
 commandActive=false;
 std::fill(commandReserved,commandReserved+N,0);
 if(commandFrom>=0){
  bool matching=o->previous&&o->previous->action==commandAction;
  bool received=matching&&o->previous->valid;
  if(!o->previous&&mine(commandTo)&&army(commandTo)>=commandExpected&&
     (!mine(commandFrom)||army(commandFrom)<commandBefore-commandMoving/2))received=true;
  if(received){
   if(commandColumn==commandFrom){
    int left=mine(commandFrom)?army(commandFrom):0;
    if(commandTravels||!commandHalf||army(commandTo)>left)commandColumn=commandTo;
   }
   commandFrom=-1;
  }else if(matching||tick-commandIssued>2)commandFrom=-1;
 }
 if(commandColumn>=0&&(!mine(commandColumn)||commandColumn==g||army(commandColumn)<8||tick-commandBorn>100)){
  commandColumn=-1;commandGoal=-1;commandReturn=false;
 }
 if(commandColumn>=0&&columnClaimed(commandColumn)){
  commandColumn=-1;commandGoal=-1;commandFrom=-1;commandReturn=false;
 }
 bool unknownKing=false;
 for(int e=0;e<o->players;e++)if(e!=p&&o->alive[e]&&enemyGeneral[e]<0)unknownKing=true;
 if(!unknownKing){commandColumn=-1;commandGoal=-1;commandReturn=false;return;}
 if(commandColumn<0&&tick>=120){
  int available=-1;
  for(int c=0;c<N;c++)if(mine(c)&&st[c]==0&&ready(c)>=24&&ed[c]<=10&&gd[c]>=4&&
      !columnClaimed(c)&&(available<0||ready(c)>ready(available)))available=c;
  if(available<0)return;
  commandColumn=available;commandGoal=-1;commandBorn=tick;commandProgress=tick;
  // This floor persists across frontier waypoints. Successive replans must
  // not ratchet a 50-troop operation down into a handful of isolated raiders.
  commandMinimum=std::max(12,ready(commandColumn)*2/3);
  commandNeed=ready(commandColumn);
 }
}

bool commandAllows(const Program &q)const {
 if(!commandActive||q.kind==26)return true;
 auto x=q.orders[q.cursor];
 if(x.s<0||x.s>=N)return false;
 if(!commandReserved[x.s])return true;
 if(x.s==commandNext.s&&x.t==commandNext.t&&x.half==commandNext.half)return true;
 // Imminent defence, a real field engagement, and an identified productive
 // or elimination target may redirect the operation. Routine expansion and
 // neutral purchases may use other armies, not its already assigned donors.
 if(q.kind==3||q.kind==4||q.kind==6||q.kind==9||q.kind==13||q.kind==17||q.kind==20||q.kind==21)return true;
 if(q.goal>=0&&st[q.goal]>0&&st[q.goal]<3&&own[q.goal]>=0&&own[q.goal]!=p)return true;
 return false;
}

bool homeCoveredWithoutColumn(){
 std::fill(commandHomeDistance,commandHomeDistance+N,short(-1));
 int head=0,tail=1;queue[0]=g;commandHomeDistance[g]=0;
 int other=0;
 while(head<tail){
  int u=queue[head++],d=commandHomeDistance[u];
  if(u!=g&&u!=commandColumn){
   bool half=false;if(canDepart(u,half))other=std::max(other,std::max(0,moveArmy(u,half)-d+1));
  }
  if(d>=6||(u!=g&&reserve[u]>1))continue;
  for(int k=0;k<4;k++){
   int v=nb[u][k];if(v<0||!mine(v)||commandHomeDistance[v]>=0)continue;
   commandHomeDistance[v]=d+1;queue[tail++]=v;
  }
 }
 // An army already credited to exposed-home insurance is a home reserve,
 // not simultaneously an available expedition. The need is measured before
 // subtracting this column; otherwise a one-step recall immediately makes
 // the dispatcher send the same troops out again.
 int need=std::max(exposedHomeNeed,readinessTarget());
 return ready(g)+other>=need;
}

double commandPathValue(int goal,int *path,int &length){
 length=0;
 for(int c=goal;c!=commandColumn&&c>=0&&length<25;c=opParent[c])path[length++]=c;
 if(length<1||length>24||opForce[goal]<commandMinimum)return -1e30;
 std::fill(expeditionVision,expeditionVision+N,0);
 double mass=0;int losses=0,land=0;
 for(int j=length-1;j>=0;j--){
  int c=path[j];
  if(!mine(c)){losses+=rememberedDefense(c)+1;land++;}
  for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){
   int x=cellX[c]+dx,y=cellY[c]+dy;if(x<0||x>=W||y<0||y>=H)continue;
   int v=y*W+x;if(expeditionVision[v])continue;
   expeditionVision[v]=1;mass+=kingInformation[v];
  }
 }
 if(mass<=0)return -1e30;
 // A concrete reconnaissance tour values new information once. Ordinary
 // captures pay their production value, without a second enemy-vision prize.
 return (informationPrice*mass+land)/(length+1.+losses*.12);
}

void commandPrograms(){
 observeCommand();
 if(commandColumn<0||commandFrom>=0||baselineDanger>1||!planningTimeLeft())return;
 int root=commandColumn;bool half=false;
 bool independent=homeCoveredWithoutColumn();
 if(!independent&&(commandReturn||commandHomeDistance[root]>=0)){
  commandReturn=true;return;
 }
 if(independent)commandReturn=false;
 if(!canDepart(root,half))return;
 Program chosen;chosen.kind=26;chosen.root=root;chosen.born=tick;
 if(moveArmy(root,half)<commandMinimum+4){
  // Refill the same force; do not spend its surviving strength on another
  // unrelated purchase merely because the original tour became unaffordable.
  scheduleTree(root,8);int budget=-1,best=ready(root);
  for(int k=1;k<=8;k++)if(schedule[root][k]>best+3){
   budget=k;best=schedule[root][k];if(best>=commandNeed)break;
  }
  if(budget>=0){
   extractSchedule(root,budget,chosen);chosen.goal=root;
   chosen.value=std::min(8.,3.+double(best-ready(root))/(budget+2.));
  }else{
   // A column outside its collection neighbourhood can return to a real
   // donor. The entire owned route and every protected transfer are priced
   // by taskRoutes; no remote aggregate army is promised as instant supply.
   taskRoutes(root,moveArmy(root,half),true,true);
   int goal=-1;double priority=0;
   for(int c=0;c<N;c++)if(mine(c)&&c!=root&&ready(c)>=8&&opSteps[c]>0&&opSteps[c]<=24&&opForce[c]>=commandNeed){
    double value=std::min(commandNeed,opForce[c]-ready(root))/(1.+routeCost[c]);
    if(value>priority){priority=value;goal=c;}
   }
   if(goal<0)return;
   int path[25],n=0;for(int c=goal;c!=root&&c>=0&&n<25;c=opParent[c])path[n++]=c;
   if(n<1||n>24)return;
   chosen.goal=goal;chosen.value=std::min(8.,3.+priority);
   int u=root;for(int j=n-1;j>=0;j--){int v=path[j];chosen.orders[chosen.count++]={u,v,j==n-1?half:opHalf[v]};u=v;}
  }
 }else{
  taskRoutes(root,moveArmy(root,half),true,false,true);
  int goals[12],count=0;double scores[12];
  for(int c=0;c<N;c++)if(opSteps[c]>0&&opSteps[c]<=24&&opForce[c]>=commandMinimum&&informationGain[c]>0){
   double value=informationGain[c]/(1.+routeCost[c]);
   int j=count;if(j==12){j=11;if(value<=scores[j])continue;}else count++;
   while(j>0&&value>scores[j-1]){if(j<12){goals[j]=goals[j-1];scores[j]=scores[j-1];}j--;}
   goals[j]=c;scores[j]=value;
  }
  int selected=-1;double best=-1e30;int bestPath[25],bestLength=0;
  for(int j=-1;j<count;j++){
   int goal=j<0?commandGoal:goals[j];
   if(goal<0||opSteps[goal]<1||opSteps[goal]>24||informationGain[goal]<=0)continue;
   int path[25],length=0;double value=commandPathValue(goal,path,length);
   if(goal==commandGoal)value*=1.25;
   if(value>best){best=value;selected=goal;bestLength=length;std::copy(path,path+length,bestPath);}
  }
  if(selected<0)return;
  chosen.goal=selected;
  chosen.value=std::max(3.5,std::min(8.,best+.15*std::min(24,tick-commandProgress)));
  int u=root;
  for(int j=bestLength-1;j>=0;j--){int v=bestPath[j];chosen.orders[chosen.count++]={u,v,j==bestLength-1?half:opHalf[v]};u=v;}
 }
 if(!chosen.count||!permitted(chosen.orders[0])||landingCanBeDenied(chosen.orders[0]))return;
 commandGoal=chosen.goal==root?commandGoal:chosen.goal;
 commandNext=chosen.orders[0];commandActive=true;
 commandReserved[root]=1;
 for(int j=0;j<chosen.count;j++)if(mine(chosen.orders[j].s))commandReserved[chosen.orders[j].s]=1;
 offer(chosen);
}

void recordCommand(const Program &q,const Order &x,std::int64_t action){
 if(commandColumn<0)return;
 if(q.kind==26)commandProgress=tick;
 if(x.s!=commandColumn)return;
 commandFrom=x.s;commandTo=x.t;commandIssued=tick;commandAction=action;commandHalf=x.half;
 commandTravels=q.kind==26;
 if(q.kind==3||q.kind==6)commandReturn=true;
 commandBefore=army(x.s);commandMoving=moveArmy(x.s,x.half);
 commandExpected=std::max(1,commandMoving+(mine(x.t)?army(x.t):-rememberedDefense(x.t)));
}
