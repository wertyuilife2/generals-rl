// Reinforcement is a scheduled operation, not an assumed garrison. A city
// may request only troops that can reach it before the observed attack, and
// shares the existing retention budget with all other defended cities.
int cityDefenseBudget(int root)const {
 int mobile=std::max<int64_t>(0,o->total_army[p]-o->total_land[p]);
 int available=std::max(12,int(mobile*.25));
 for(int c=0;c<N;c++)if(c!=root&&mine(c)&&st[c]==1&&reserve[c]>1)
  available-=reserve[c];
 return std::max(0,available);
}

int cityDefenseNeed(int city)const {
 if(city<0||!mine(city)||st[city]!=1||zoneETA[city]>6)return 0;
 int need=zoneThreat[city]+2;
 if(need<=ready(city)+futureGrowth(zoneETA[city])||need>cityDefenseBudget(city))return 0;
 return need;
}

double cityDefenseValue(const Program &plan){
 int root=plan.root,need=cityDefenseNeed(root);
 if(!need||plan.goal!=root)return -1e30;
 Program rest;rest.root=rest.goal=root;rest.kind=26;
 for(int j=plan.cursor;j<plan.count;j++)rest.orders[rest.count++]=plan.orders[j];
 // A one-order relief can also retake a city captured by an earlier seat
 // in this joint round. The common interception check validates the landing.
 int deadline=std::max(1,zoneETA[root]-1);
 if(!rest.count||rest.count>deadline)return -1e30;
 int arrived=collectionArrival(rest,root);
 if(arrived<need)return -1e30;
 int committed=std::max(0,arrived-ready(root));
 double protectedIncome=.5*capitalHorizon+(gd[root]<=6?18.:0.);
 return std::min(12.,protectedIncome/(rest.count+2.+committed/25.));
}

void cityDefensePrograms(){
 if(tick<80||baselineDanger>1)return;
 int goals[3],count=0;double priority[3];
 for(int c=0;c<N;c++){
  int need=cityDefenseNeed(c);if(!need)continue;
  double value=(40.+(gd[c]<=6?18.:0.))/(need-ready(c)+12.)/(zoneETA[c]+1.);
  int j=count;if(j==3){j=2;if(value<=priority[j])continue;}else count++;
  while(j>0&&value>priority[j-1]){if(j<3){goals[j]=goals[j-1];priority[j]=priority[j-1];}j--;}
  goals[j]=c;priority[j]=value;
 }
 for(int gi=0;gi<count;gi++){
  if(!planningTimeLeft())break;
  int root=goals[gi],need=cityDefenseNeed(root),budget=std::max(1,zoneETA[root]-1);
  scheduleTree(root,budget);Program best;
  for(int k=1;k<=budget;k++)if(schedule[root][k]>=need){
   Program q;q.kind=26;q.root=q.goal=root;q.born=tick;extractSchedule(root,k,q);
   if(collectionArrival(q,root)<need)continue;
   // Do not move an entire independent field column when half can fund
   // the same defense. Re-simulate the shared ledger after every change.
   for(int j=0;j<q.count;j++)if(!q.orders[j].half){
    q.orders[j].half=true;
    if(collectionArrival(q,root)<need)q.orders[j].half=false;
   }
   q.value=cityDefenseValue(q);
   if(q.value>best.value&&q.count&&permitted(q.orders[0]))best=q;
  }
  offer(best);
 }
}
