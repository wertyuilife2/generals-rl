// Coordinate tactical emergencies, economic routes and persistent operations.
Program activeOperation;
int operationIssued=-1000,operationProgress=-1000,operationSource=-1;
int operationBefore=0,operationSent=0,operationDestination=-1;
std::int64_t operationAction=0;
bool operationPending=false;
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
 int need=24;
 for(int e=0;e<o->players;e++)if(e!=p&&o->alive[e])need=std::max(need,int((o->total_army[e]-o->total_land[e])*.35));
 for(int c=0;c<N;c++)if(o->visible[c]&&own[c]>=0&&own[c]!=p&&
    std::abs(cellX[c]-cellX[plan.root])+std::abs(cellY[c]-cellY[plan.root])<=10)need=std::max(need,army(c)+12);
 need=std::min(need,int((o->total_army[p]-o->total_land[p])*.65));
 int useful=mobileAssemblyCredit(rest,plan.root,simulated[plan.root],need);
 double value=.70*useful/(rest.count+2.)/(1.+.08*ed[plan.root]);
 value=9.*(-std::expm1(-value/9.));
 if(ready(plan.root)<need&&simulated[plan.root]>=need)value+=std::sqrt(double(useful)/std::max(1,need));
 return value;
}

void resumeOperation(){
 if(activeOperation.kind==0)return;
 if(activeOperation.kind==16&&expeditionCredit<1)return;
 if(operationPending){
  bool matching=o->previous&&o->previous->action==operationAction;
  bool received=matching&&o->previous->valid;
  if(!o->previous&&mine(operationDestination)&&
     (!mine(operationSource)||army(operationSource)<operationBefore-operationSent/2))received=true;
  if(received){operationPending=false;operationProgress=tick;}
  else if(matching||tick-operationIssued>2){activeOperation=Program();operationPending=false;return;}
  else return;
 }
 if(activeOperation.cursor>=activeOperation.count){
  if(activeOperation.kind==11&&mine(activeOperation.root)){rallyCell=activeOperation.root;rallyTick=tick;}
  if(activeOperation.kind==14&&mine(activeOperation.goal)){rallyCell=activeOperation.goal;rallyTick=tick;}
  activeOperation=Program();return;
 }
 bool expired=tick-activeOperation.born>48||tick-operationProgress>10;
 bool explored=(activeOperation.kind==10||activeOperation.kind==16)&&intelGain[activeOperation.goal]<1.;
 auto &next=activeOperation.orders[activeOperation.cursor];
 if(expired||explored||!mine(next.s)||blocked(next.t)){activeOperation=Program();return;}
 if(activeOperation.kind==14?extractionSafe(activeOperation):permitted(next)){
  Program q=activeOperation;
  if(q.kind==11)q.value=remainingAssemblyValue(q);
  q.value*=1.09;offer(q);
 }
}

void rememberOperation(const Program &chosen,const Order &action,std::int64_t encoded){
 if(chosen.kind!=10&&chosen.kind!=11&&chosen.kind!=12&&chosen.kind!=14&&chosen.kind!=15&&chosen.kind!=16&&chosen.kind!=19)return;
 activeOperation=chosen;operationIssued=tick;operationSource=action.s;
 operationDestination=action.t;operationBefore=army(action.s);operationSent=moveArmy(action.s,action.half);
 operationAction=encoded;operationPending=true;
 if(chosen.born==tick)operationProgress=tick;
}

std::int64_t decideV8(const ArenaObservation *obs){
 if(obs->width*obs->height>MAXN||obs->general_pos<0)return 0;
 update(obs);updateExpeditionCredit();assetHistory();strategicMode();pressureMap();intelligenceMap();supplyMap();buildKingRelief();buildAttacks();baselineDanger=dangerAfter(nullptr,homeIncoming);proposalCount=0;
 resumeOperation();
 // Finish a concrete, validated program unless new information or a better
 // opportunity justifies interruption. Skipped online ticks invalidate it.
 bool received=o->previous?(o->previous->valid&&o->previous->action==missionAction):
    (missionReceiptCell>=0&&mine(missionReceiptCell)&&army(missionReceiptCell)>=missionReceiptForce);
 bool stillNeeded=(mission.kind!=3||baselineDanger>0||raidNeed>army(g))&&(mission.kind!=6||army(g)<std::min(int(o->total_army[p]*.45),int(homeIncoming[8]*1.35)+6));
 if(mission.kind<10&&missionTick==tick-1&&mission.cursor<mission.count&&tick-mission.born<=32&&received&&stillNeeded&&
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
 Candidate defense=defend(),forecast=forecastDefend();
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
 campaignPrograms(income);
 routePlan(income); // Registers the full column program, including collections.
 if(tick>=48)logisticsPrograms(income);
 readinessProgram();
 deploymentPrograms();
 homeCounterPrograms();
 interceptionPrograms();
 fieldEngagementPrograms();
 coordinatedAssaultPrograms();
 // Small-scout trials are archived; no routine scouting order is forced.
 reconnaissancePrograms();
 // Expedition proposals remain archived for isolated research.
 mobilizationPrograms();
 extractionPrograms();
 unlockGarrisonPrograms();
 // City battle search disabled in this ablation.

 // A common post-action safety gate covers every mode and every player count.
 // Safe alternatives are preferred before comparing economic benefits.
 double bestScore=-.01-baselineDanger;int best=-1;
 for(int j=0;j<proposalCount;j++){
  double score=proposalScore[j];
  if(score>bestScore){bestScore=score;best=j;}
 }
 if(best<0){mission=Program();missionTick=tick;return 0;}
 Program chosen=proposals[best];Order action=chosen.orders[chosen.cursor++];
 if(chosen.kind==16||chosen.kind==18)expeditionCredit-=1;
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
