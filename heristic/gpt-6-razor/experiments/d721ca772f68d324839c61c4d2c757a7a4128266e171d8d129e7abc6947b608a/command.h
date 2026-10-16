// A field operation owns its army until it reaches an information objective,
// is redirected by a military emergency, or loses the ability to continue.
// The objective survives individual moves; a neighbouring cheap tile is not
// a new operation. Every remembered location comes from this seat's receipts.
int commandColumn=-1,commandGoal=-1,commandBorn=-1000,commandProgress=-1000;
int commandMinimum=0,commandNeed=0;
int commandFrom=-1,commandTo=-1,commandIssued=-1000,commandExpected=0;
int commandBefore=0,commandMoving=0;
bool commandHalf=false,commandActive=false;
std::int64_t commandAction=0;
unsigned char commandReserved[MAXN]{};
Order commandNext;

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
    if(!commandHalf||army(commandTo)>left)commandColumn=commandTo;
   }
   commandFrom=-1;
  }else if(matching||tick-commandIssued>2)commandFrom=-1;
 }
 if(commandColumn>=0&&(!mine(commandColumn)||commandColumn==g||army(commandColumn)<8||tick-commandBorn>100)){
  commandColumn=-1;commandGoal=-1;
 }
 bool unknownKing=false;
 for(int e=0;e<o->players;e++)if(e!=p&&o->alive[e]&&enemyGeneral[e]<0)unknownKing=true;
 if(!unknownKing){commandColumn=-1;commandGoal=-1;return;}
 if(commandColumn<0&&tick>=120&&mainColumn>=0&&gd[mainColumn]>=4){
  commandColumn=mainColumn;commandGoal=-1;commandBorn=tick;commandProgress=tick;
  // This floor persists across frontier waypoints. Successive replans must
  // not ratchet a 50-troop operation down into a handful of isolated raiders.
  commandMinimum=std::max(12,ready(commandColumn)*2/3);
  commandNeed=ready(commandColumn);
 }
}

bool commandAllows(const Program &q)const {
 if(!commandActive||q.kind==26)return true;
 auto x=q.orders[q.cursor];
 if(!commandReserved[x.s])return true;
 if(x.s==commandNext.s&&x.t==commandNext.t&&x.half==commandNext.half)return true;
 // Imminent defence, a real field engagement, and an identified productive
 // or elimination target may redirect the operation. Routine expansion and
 // neutral purchases may use other armies, not its already assigned donors.
 if(q.kind==3||q.kind==4||q.kind==6||q.kind==9||q.kind==13||q.kind==17||q.kind==20||q.kind==21)return true;
 if(q.goal>=0&&st[q.goal]>0&&st[q.goal]<3&&own[q.goal]>=0&&own[q.goal]!=p)return true;
 return false;
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
 if(!canDepart(root,half))return;
 Program chosen;chosen.kind=26;chosen.root=root;chosen.born=tick;
 if(moveArmy(root,half)<commandMinimum+4){
  // Refill the same force; do not spend its surviving strength on another
  // unrelated purchase merely because the original tour became unaffordable.
  scheduleTree(root,8);int budget=-1,best=ready(root);
  for(int k=1;k<=8;k++)if(schedule[root][k]>best+3){
   budget=k;best=schedule[root][k];if(best>=commandNeed)break;
  }
  if(budget<0)return;
  extractSchedule(root,budget,chosen);chosen.goal=root;
  chosen.value=std::min(8.,3.+double(best-ready(root))/(budget+2.));
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
 commandBefore=army(x.s);commandMoving=moveArmy(x.s,x.half);
 commandExpected=std::max(1,commandMoving+(mine(x.t)?army(x.t):-rememberedDefense(x.t)));
}
