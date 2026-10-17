// Joint mobile defense of a compact group of assets. All coverage estimates
// share one actual collection program. This is a bounded visible-column
// response, not a guarantee about hidden armies or every possible opponent.
int regionEnemy=-1,regionCount=0,regionAssets[3],regionNeed[3];
int regionEnemyDistance[MAXN],regionDistance[3][MAXN];
int regionTroops[MAXN],regionSavedReserve[3];
bool regionCovered=false;

void buildRegion(){
 regionEnemy=-1;regionCount=0;regionCovered=false;
 if(tick<120||aliveCount!=2||baselineDanger>1)return;
 // Multiple sizeable observed columns need the existing multi-threat
 // emergency handler. One response force is not promised to both of them.
 int second=0;
 for(int c=0;c<N;c++)if(o->visible[c]&&own[c]>=0&&own[c]!=p&&st[c]==0&&
     gd[c]<=14&&ready(c)>=24){
  if(regionEnemy<0||ready(c)>ready(regionEnemy)){
   if(regionEnemy>=0)second=ready(regionEnemy);regionEnemy=c;
  }else second=std::max(second,ready(c));
 }
 if(regionEnemy<0||second>0){regionEnemy=-1;return;}
 bfs(regionEnemy,regionEnemyDistance);
 regionAssets[0]=g;regionCount=1;
 int cities[2]={-1,-1};double rank[2]={-1,-1};
 for(int c=0;c<N;c++)if(mine(c)&&st[c]==1&&gd[c]<=8&&regionEnemyDistance[c]<=10){
  double value=1./(1.+regionEnemyDistance[c]);
  int j=value>rank[0]?0:value>rank[1]?1:-1;if(j<0)continue;
  if(j==0){cities[1]=cities[0];rank[1]=rank[0];}cities[j]=c;rank[j]=value;
 }
 for(int c:cities)if(c>=0)regionAssets[regionCount++]=c;
 if(regionCount<2){regionEnemy=-1;regionCount=0;return;}
 for(int i=0;i<regionCount;i++){
  bfs(regionAssets[i],regionDistance[i],true);
  // Travel costs at least one troop per edge. No friendly field garrison
  // is subtracted here: that same army may be collected into the defender.
  regionNeed[i]=std::max(0,ready(regionEnemy)-regionEnemyDistance[regionAssets[i]]+2);
  if(i==0&&homeKnownTo[own[regionEnemy]]){
   int enemy=own[regionEnemy],pulses=(tick+6)/50-tick/50;
   int field=int(std::min<int64_t>(300,std::max<int64_t>(0,
      o->total_army[enemy]-o->total_land[enemy]+pulses*o->total_land[enemy])));
   int insurance=int(raidValue*std::min(field*.40,double(o->total_army[p])*.30));
   regionNeed[i]=std::max(regionNeed[i],insurance);
  }
 }
 for(int c=0;c<N;c++)regionTroops[c]=mine(c)?ready(c):0;
 regionCovered=regionHasPost(regionTroops,0);
}

bool regionHasPost(const int *troops,int elapsed)const {
 if(regionEnemy<0||regionCount<2)return false;
 // One concrete post must cover every asset; each potential response is
 // conditional on which asset the single observed column approaches.
 for(int c=0;c<N;c++)if(mine(c)&&troops[c]>=24){
  if(troops[c]<ready(regionEnemy)-regionEnemyDistance[c]+2)continue;
  bool covers=true;
  for(int i=0;i<regionCount;i++){
   int asset=regionAssets[i],d=regionDistance[i][c];
   int deadline=regionEnemyDistance[asset]-int(own[regionEnemy]<p);
   if(d>4||d+elapsed>deadline){covers=false;break;}
   int cover=troops[asset]+(c==asset?0:std::max(0,troops[c]-d));
   if(cover<regionNeed[i]){covers=false;break;}
  }
  if(covers)return true;
 }
 return false;
}

bool regionPlanSound(const Program &q){
 if(regionEnemy<0||q.cursor>=q.count||q.count-q.cursor>8||!mine(q.root))return false;
 int steps=q.count-q.cursor;
 if(!legalOrder(q.orders[q.cursor])||landingCanBeDenied(q.orders[q.cursor])||
    dangerAfter(&q.orders[q.cursor])>std::max(1.,baselineDanger))return false;
 for(int c=0;c<N;c++)regionTroops[c]=mine(c)?ready(c):0;
 for(int j=q.cursor;j<q.count;j++){
  auto x=q.orders[j];int step=j-q.cursor;
  if(!mine(x.s)||!mine(x.t)||direction(x.s,x.t)<0)return false;
  // Collection finishes before the observed column can occupy a transport
  // square. Close contested assemblies remain the tactical handler's job.
  if(std::min(regionEnemyDistance[x.s],regionEnemyDistance[x.t])<=
       steps+int(own[regionEnemy]<p))return false;
  if(step)for(int c=0;c<N;c++)if(mine(c))regionTroops[c]+=grows(c,actionTurn(step+1));
  if(zoneETA[x.s]<=step+int(own[regionEnemy]<p)&&zoneThreat[x.s]>regionTroops[x.s])return false;
  int sent=x.half?regionTroops[x.s]/2:regionTroops[x.s]-1;
  if(sent<1)return false;
  bool asset=false;for(int i=0;i<regionCount;i++)asset|=x.s==regionAssets[i];
  if(!asset&&regionTroops[x.s]-sent<reserve[x.s])return false;
  regionTroops[x.s]-=sent;regionTroops[x.t]+=sent;
  if(zoneETA[x.t]<=step+1&&zoneThreat[x.t]>regionTroops[x.t])return false;
 }
 // The final post, not a sum of independently promised donors, funds the
 // complete return trip. Subsequent observations revalidate the program.
 int force=regionTroops[q.root];
 if(force<24||force<ready(regionEnemy)-regionEnemyDistance[q.root]+2)return false;
 for(int i=0;i<regionCount;i++){
  int asset=regionAssets[i],d=regionDistance[i][q.root];
  if(d>4||steps+d>regionEnemyDistance[asset]-int(own[regionEnemy]<p))return false;
  int cover=regionTroops[asset]+(asset==q.root?0:std::max(0,force-d));
  if(cover<regionNeed[i])return false;
 }
 return true;
}

void regionalPrograms(){
 if(regionEnemy<0||regionCovered)return;
 int roots[4],count=0;double priorities[4];
 for(int c=0;c<N;c++)if(mine(c)&&c!=g){
  bool close=true;double proximity=0;
  for(int i=0;i<regionCount;i++){
   int d=regionDistance[i][c];
   if(d>4||regionEnemyDistance[regionAssets[i]]-d<=2){close=false;break;}
   proximity+=1./(d+1.);
  }
  if(!close)continue;
  double stock=ready(c);
  for(int u=0;u<N;u++)if(mine(u)&&u!=c&&ready(u)>=4){
   int d=std::abs(cellX[c]-cellX[u])+std::abs(cellY[c]-cellY[u]);
   if(d<=4)stock+=(ready(u)-1.)/(d+1.);
  }
  double priority=stock*proximity;
  int j=count;if(j==4){j=3;if(priority<=priorities[j])continue;}else count++;
  while(j>0&&priority>priorities[j-1]){
   if(j<4){roots[j]=roots[j-1];priorities[j]=priorities[j-1];}j--;
  }
  roots[j]=c;priorities[j]=priority;
 }
 for(int ri=0;ri<count;ri++){
  if(!planningTimeLeft())break;
  int root=roots[ri],budget=8;
  for(int i=0;i<regionCount;i++)budget=std::min(budget,
    regionEnemyDistance[regionAssets[i]]-regionDistance[i][root]-int(own[regionEnemy]<p));
  if(budget<1)continue;
  // The existing tree reconstructs exact disjoint transfers. Temporarily
  // expose only the region's own asset reserves to that collection; no
  // other scorer observes this override, and all originals are restored.
  for(int i=0;i<regionCount;i++){
   int c=regionAssets[i];regionSavedReserve[i]=reserve[c];reserve[c]=1;
  }
  scheduleTree(root,budget);
  Program choices[9];
  for(int k=1;k<=budget;k++)if(schedule[root][k]>0){
   auto &q=choices[k];q.kind=27;q.root=q.goal=root;q.born=tick;
   extractSchedule(root,k,q);
  }
  for(int i=0;i<regionCount;i++)reserve[regionAssets[i]]=regionSavedReserve[i];
  for(int k=1;k<=budget;k++){
   auto &q=choices[k];if(!q.count||!regionPlanSound(q))continue;
   int finalForce=regionTroops[root];
   int useful=mobileAssemblyCredit(q,root,finalForce,finalForce);
   if(useful<4)continue;
   q.value=34.+3.*(regionCount-1)+std::min(12.,double(useful)/(k+1.));
   offer(q);break;
  }
 }
}

double regionDepartureCost(const Program &q){
 if(!regionCovered||regionEnemy<0||q.kind==3||q.kind==4||q.kind==9||
     q.kind==14||q.kind==17||q.kind==20||q.kind==21||q.kind==27)return 0;
 // Price the first actual order. If it destroys the existing joint response,
 // a new speculative multi-turn route cannot spend that protection for free.
 auto x=q.orders[q.cursor];
 for(int c=0;c<N;c++)regionTroops[c]=mine(c)?ready(c):0;
 int sent=moveArmy(x.s,x.half);regionTroops[x.s]-=sent;
 if(mine(x.t))regionTroops[x.t]+=sent;
 // A captured square may be useful later, but is not silently inserted into
 // the old owned return tree as a free shortcut.
 if(regionHasPost(regionTroops,1))return 0.;
 return 18.+6.*(regionCount-1);
}
