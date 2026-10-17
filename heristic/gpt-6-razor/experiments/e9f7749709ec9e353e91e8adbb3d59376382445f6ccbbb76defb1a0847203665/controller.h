// Coordinate tactical emergencies, economic routes and persistent operations.
struct OperationTask {
 Program plan;
 int issued=-1000,progress=-1000,source=-1,destination=-1;
 int before=0,sent=0,expected=0;
 std::int64_t action=0;
 bool pending=false;
};
OperationTask operationTasks[4];
int operationSerial=0;
int rallyCell=-1,rallyTick=-1000;

double remainingAssemblyValue(const Program &plan){
 int simulated[MAXN];for(int c=0;c<N;c++)simulated[c]=ready(c);
 Program rest;rest.root=plan.root;rest.goal=plan.goal;
 for(int j=plan.cursor;j<plan.count;j++){
  auto x=plan.orders[j];if(!mine(x.s)||!mine(x.t))return -1e30;
  if(reserve[x.s]>1)x.half=true;
  int sent=x.half?simulated[x.s]/2:simulated[x.s]-1;
  if(sent<1||simulated[x.s]-sent<reserve[x.s])return -1e30;
  simulated[x.s]-=sent;simulated[x.t]+=sent;rest.orders[rest.count++]=x;
 }
 int need=assemblyRequirement(plan.root);
 int useful=mobileAssemblyCredit(rest,plan.root,simulated[plan.root],need);
 double value=.70*useful/(rest.count+2.)/(1.+.08*ed[plan.root]);
 value=9.*(-std::expm1(-value/9.));
 if(ready(plan.root)<need&&simulated[plan.root]>=need)value+=std::sqrt(double(useful)/std::max(1,need));
 return value;
}

void resumeOperation(){
 for(auto &task:operationTasks){
 auto &plan=task.plan;
 if(plan.kind==0)continue;
 if(task.pending){
  bool matching=o->previous&&o->previous->action==task.action;
  bool received=matching&&o->previous->valid;
  if(!o->previous&&mine(task.destination)&&army(task.destination)>=task.expected&&
     (!mine(task.source)||army(task.source)<task.before-task.sent/2))received=true;
  if(received){task.pending=false;task.progress=tick;}
  else if(matching||tick-task.issued>2){task=OperationTask();continue;}
  else continue;
 }
 if(plan.cursor>=plan.count){
  if((plan.kind==11||plan.kind==25||plan.kind==27)&&mine(plan.root)){rallyCell=plan.root;rallyTick=tick;}
  if(plan.kind==14&&mine(plan.goal)){rallyCell=plan.goal;rallyTick=tick;}
  task=OperationTask();continue;
 }
 if(plan.kind==23&&mine(plan.goal)){task=OperationTask();continue;}
 if(plan.kind==24&&expeditionCredit<1)continue;
 bool expired=tick-plan.born>64||tick-task.progress>20;
 bool explored=(plan.kind==10||plan.kind==24)&&intelGain[plan.goal]<1.;
 auto &next=plan.orders[plan.cursor];
 if(plan.kind==23&&next.t==plan.goal&&st[next.t]==1&&
    assetFlips[next.t]>=2&&tick-assetLostAt[next.t]<40&&
    moveArmy(next.s,next.half)-rememberedDefense(next.t)<zoneThreat[next.t]+8){
  task=OperationTask();continue;
 }
 if(expired||explored||!mine(next.s)||blocked(next.t)){task=OperationTask();continue;}
 if(plan.kind==27?regionPlanSound(plan):(plan.kind==14?extractionSafe(plan):permitted(next))){
  Program q=plan;
  if(q.kind==11)q.value=remainingAssemblyValue(q);
  if(q.kind==16)q.value=expeditionValue(q);
  q.value*=1.09;offer(q);
 }
 }
}

void rememberOperation(const Program &chosen,const Order &action,std::int64_t encoded){
 int slot=-1;
 for(int i=0;i<4;i++){
  auto &task=operationTasks[i];auto &q=task.plan;if(!q.kind)continue;
  if(chosen.taskId&&q.taskId==chosen.taskId){slot=i;continue;}
  // Only an executed withdrawal spends another task's donor. Merely
  // proposing a plan, or supplying that donor, cannot invalidate its task.
  bool spent=task.pending&&task.destination==action.s;
  for(int j=q.cursor;j<q.count;j++)if(q.orders[j].s==action.s)spent=true;
  if(spent)task=OperationTask();
 }
 bool persistent=chosen.kind==1||chosen.kind==2||chosen.kind==8||chosen.kind==10||
  chosen.kind==11||chosen.kind==12||chosen.kind==14||chosen.kind==15||chosen.kind==16||
  chosen.kind==19||chosen.kind==22||chosen.kind==23||chosen.kind==24||chosen.kind==25||chosen.kind==27;
 if(!persistent||chosen.count<=1)return;
 if(slot<0)for(int i=0;i<4;i++)if(!operationTasks[i].plan.kind){slot=i;break;}
 if(slot<0){
  slot=0;for(int i=1;i<4;i++)if(operationTasks[i].progress<operationTasks[slot].progress)slot=i;
 }
 auto &task=operationTasks[slot];int progress=chosen.taskId?task.progress:tick;
 task=OperationTask();task.plan=chosen;
 if(!task.plan.taskId)task.plan.taskId=++operationSerial;
 task.progress=progress;task.issued=tick;task.source=action.s;task.destination=action.t;
 task.before=army(action.s);task.sent=moveArmy(action.s,action.half);
 task.expected=std::max(1,task.sent+(mine(action.t)?army(action.t):-rememberedDefense(action.t)));
 task.action=encoded;task.pending=true;
}

std::int64_t decideV8(const ArenaObservation *obs){
 if(obs->width*obs->height>MAXN||obs->general_pos<0)return 0;
 planningStart=std::chrono::steady_clock::now();
 update(obs);updateExpeditionCredit();assetHistory();observeCityResponses();observeColumns();buildOpponentAssembly();strategicMode();pressureMap();assignMainColumn();intelligenceMap();supplyMap();buildKingRelief();buildCityRelief();buildAttacks();baselineDanger=dangerAfter(nullptr,homeIncoming);proposalCount=0;buildRegion();
 resumeOperation();
 if(planningTimeLeft())regionalPrograms();
 bool regionalTask=false;for(int j=0;j<proposalCount;j++)regionalTask|=proposals[j].kind==27;
 // Finish a concrete, validated program unless new information or a better
 // opportunity justifies interruption. Skipped online ticks invalidate it.
 bool received=o->previous?(o->previous->valid&&o->previous->action==missionAction):
    (missionReceiptCell>=0&&mine(missionReceiptCell)&&army(missionReceiptCell)>=missionReceiptForce);
 bool stillNeeded=(mission.kind!=3||baselineDanger>0||raidNeed>army(g))&&(mission.kind!=6||ready(g)<readinessTarget());
 if((mission.kind==3||mission.kind==6)&&missionTick==tick-1&&mission.cursor<mission.count&&tick-mission.born<=32&&received&&stillNeeded&&
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
 Candidate defense,forecast;
 if(!regionalTask&&!regionCovered){defense=defend();forecast=forecastDefend();}
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
 if(planningTimeLeft())campaignPrograms(income);
 if(planningTimeLeft())routePlan(income); // Full program, including collections.
 if(tick>=48&&planningTimeLeft())logisticsPrograms(income);
 if(!regionalTask&&!regionCovered)readinessProgram();
 if(planningTimeLeft())deploymentPrograms();
 homeCounterPrograms();
 interceptionPrograms();
 fieldEngagementPrograms();
 if(planningTimeLeft())coordinatedAssaultPrograms();
 if(planningTimeLeft())convergingAssaultPrograms();
 // Small-scout trials are archived; no routine scouting order is forced.
 if(planningTimeLeft())reconnaissancePrograms();
 if(planningTimeLeft())expeditionPrograms();
 if(planningTimeLeft())mobilizationPrograms();
 extractionPrograms();
 if(planningTimeLeft())unlockGarrisonPrograms();
 if(planningTimeLeft())productionInvestmentPrograms();
 if(planningTimeLeft())occupationPrograms();
 if(planningTimeLeft())informationPrograms();
 if(planningTimeLeft())frontFormationPrograms();
 if(planningTimeLeft())localBattlePrograms();

 // A common post-action safety gate covers every mode and every player count.
 // Safe alternatives are preferred before comparing economic benefits.
 double bestScore=-.01-baselineDanger;int best=-1;
 for(int j=0;j<proposalCount;j++){
  double score=proposalScore[j];
  if(score>bestScore){bestScore=score;best=j;}
 }
 if(best<0){mission=Program();missionTick=tick;return 0;}
 Program chosen=proposals[best];Order action=chosen.orders[chosen.cursor++];
 if(chosen.kind==18||chosen.kind==24)expeditionCredit-=1;
 // Continuation preference is applied afresh, never compounded every turn.
 if(chosen.born<tick)chosen.value/=1.09;
 mission=chosen;missionTick=tick;oldTarget=chosen.goal;oldSource=action.t;lastTarget=action.t;
 missionReceiptCell=action.t;
 missionReceiptForce=moveArmy(action.s,action.half)+(mine(action.t)?army(action.t):-rememberedDefense(action.t));
 missionAction=arena_move(action.s,direction(action.s,action.t),action.half);
 rememberOperation(chosen,action,missionAction);
 recordFlow(action,missionAction);
 return missionAction;
}
