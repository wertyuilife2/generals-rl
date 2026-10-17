// A bounded model of territorial raids. Collection and the observed raider
// act on one ledger; moving a donor cannot also leave it as a static screen.
// The retained paths are hypotheses, not an exhaustive opponent strategy.
struct ProvinceRaid {
 short cell[13]{};
 int count=0,force=0;
 double gain=0;
};
ProvinceRaid provinceBeam[16],provinceExpanded[64],provinceRaids[24];
int provinceCount=0,provinceEnemy=-1,provinceRoot=-1;
int provinceIndex[MAXN],provinceDepth[MAXN];
double provinceBase=0,provinceWeight[MAXN];

double provinceTileValue(int c)const {
 if(!mine(c))return 0.;
 if(c==g)return 400.;
 if(st[c]==1)return 8.;
 // The coming payout is worth more than a speculative later recapture.
 int until=50-tick%50;
 return 1.+(until<=12?1.:.5)+.015*std::min(20,ready(c));
}

void prepareProvinceRaids(){
 provinceCount=0;provinceEnemy=-1;provinceRoot=-1;provinceBase=0;
 std::fill(provinceWeight,provinceWeight+N,0.);
 std::fill(provinceDepth,provinceDepth+N,INF);
 if(aliveCount!=2||tick<110||baselineDanger>1)return;
 // A single mobile group supplies all hypotheses in this model. Distinct
 // large groups keep their existing multi-threat tactical treatment.
 int other=0;
 for(int c=0;c<N;c++)if(o->visible[c]&&own[c]>=0&&own[c]!=p&&
     st[c]==0&&ready(c)>=20&&gd[c]<=16){
  if(provinceRoot<0||ready(c)>ready(provinceRoot)){
   if(provinceRoot>=0)other=std::max(other,ready(provinceRoot));provinceRoot=c;
  }else other=std::max(other,ready(c));
 }
 if(provinceRoot<0||other>=20){provinceRoot=-1;return;}
 provinceEnemy=own[provinceRoot];
 ProvinceRaid initial;initial.cell[0]=short(provinceRoot);initial.force=ready(provinceRoot);
 provinceBeam[0]=initial;int count=1;
 for(int depth=1;depth<=12&&count;depth++){
  int generated=0;std::fill(cellRank,cellRank+N,-1e30);
  for(int j=0;j<count;j++){
   const auto &q=provinceBeam[j];int u=q.cell[depth-1];
   int moving=q.force-1+(depth>1?grows(u,actionTurn(depth)):0);if(moving<=0)continue;
   for(int k=0;k<4;k++){
    int v=nb[u][k];if(v<0||blocked(v))continue;
    bool repeat=false;for(int z=0;z<depth;z++)if(q.cell[z]==v)repeat=true;
    if(repeat)continue;
    int guard=mine(v)?ready(v):rememberedDefense(v);
    if(own[v]>=0)guard+=(st[v]>0&&st[v]<3)?actionTurn(depth)/2-actionTurn(1)/2:
                          actionTurn(depth)/50-actionTurn(1)/50;
    int force=own[v]==provinceEnemy?moving+guard:moving-guard;
    if(force<=0)continue;
    ProvinceRaid next=q;next.count=depth;next.cell[depth]=short(v);next.force=force;
    next.gain+=provinceTileValue(v);
    double rank=next.gain+.004*std::min(200,force);
    if(rank<=cellRank[v])continue;cellRank[v]=rank;
    if(generated<64)provinceExpanded[generated++]=next;
   }
  }
  std::sort(provinceExpanded,provinceExpanded+generated,[](const ProvinceRaid &a,const ProvinceRaid &b){
   return a.gain+.004*std::min(200,a.force)>b.gain+.004*std::min(200,b.force);
  });
  count=0;
  for(int j=0;j<generated&&count<16;j++){
   bool repeated=false;for(int k=0;k<count;k++)if(provinceBeam[k].cell[depth]==provinceExpanded[j].cell[depth])repeated=true;
   if(!repeated)provinceBeam[count++]=provinceExpanded[j];
  }
  if(depth==6||depth==9||depth==12){
   int saved=0;
   for(int j=0;j<count&&saved<8&&provinceCount<24;j++)if(provinceBeam[j].gain>=3.){
    provinceRaids[provinceCount++]=provinceBeam[j];saved++;
   }
  }
 }
 if(!provinceCount)return;
 for(int j=0;j<provinceCount;j++){
  const auto &r=provinceRaids[j];
  for(int k=2;k<=r.count;k++)if(mine(r.cell[k])&&r.cell[k]!=g){
   provinceWeight[r.cell[k]]+=provinceTileValue(r.cell[k])/(1.+.15*k);
   provinceDepth[r.cell[k]]=std::min(provinceDepth[r.cell[k]],k);
  }
 }
 Program wait;double loss[24];
 for(int j=0;j<provinceCount;j++){bool feasible=true;loss[j]=provinceOutcome(wait,provinceRaids[j],feasible);}
 std::sort(loss,loss+provinceCount,std::greater<double>());
 int top=std::min(4,provinceCount);for(int j=0;j<top;j++)provinceBase+=loss[j]/top;
}

double provinceOutcome(const Program &q,const ProvinceRaid &raid,bool &feasible){
 int cells[64],owners[64],troops[64],count=0;
 std::fill(provinceIndex,provinceIndex+N,-1);
 auto add=[&](int c){
  if(provinceIndex[c]>=0)return;
  int i=count++;provinceIndex[c]=i;cells[i]=c;owners[i]=own[c];
  troops[i]=mine(c)?ready(c):rememberedDefense(c);
 };
 for(int j=0;j<=raid.count;j++)add(raid.cell[j]);
 for(int j=q.cursor;j<q.count;j++){add(q.orders[j].s);add(q.orders[j].t);}
 // Every evaluated program has at most twelve transfers: the union above
 // contains at most38 cells. No full-board copy is needed per response.
 bool active=true,lostKing=false;double occupancy=0;int steps=std::max(raid.count,q.count-q.cursor);
 auto move=[&](int s,int t,bool half,int player){
  int a=provinceIndex[s],b=provinceIndex[t];
  if(owners[a]!=player)return false;
  int sent=half?troops[a]/2:troops[a]-1;if(sent<=0)return false;
  troops[a]-=sent;
  if(owners[b]==player)troops[b]+=sent;
  else if(sent>troops[b]){troops[b]=sent-troops[b];owners[b]=player;if(t==g&&player!=p)lostKing=true;}
  else troops[b]-=sent;
  return true;
 };
 for(int step=0;step<steps;step++){
  if(step)for(int j=0;j<count;j++)if(owners[j]>=0)troops[j]+=grows(cells[j],actionTurn(step+1));
  auto enemyMove=[&](){if(!active||step>=raid.count)return;
   int s=raid.cell[step],t=raid.cell[step+1];
   active=move(s,t,false,provinceEnemy)&&owners[provinceIndex[t]]==provinceEnemy;
  };
  if(provinceEnemy<p)enemyMove();
  if(q.cursor+step<q.count){
   auto x=q.orders[q.cursor+step];int s=provinceIndex[x.s],t=provinceIndex[x.t];
   // A donor cannot be used after this response removes it. Likewise, a
   // transport edge is not assumed friendly after the raider cuts it.
   if(owners[s]!=p||owners[t]!=p){feasible=false;return 1e6;}
   int sent=x.half?troops[s]/2:troops[s]-1;
   if(sent<1||troops[s]-sent<reserve[x.s]){feasible=false;return 1e6;}
   move(x.s,x.t,x.half,p);
  }
  if(provinceEnemy>p)enemyMove();
  if(lostKing)return 1e5;
  for(int j=0;j<count;j++)if(mine(cells[j])&&owners[j]!=p)occupancy+=provinceTileValue(cells[j]);
 }
 double final=0;
 for(int j=0;j<count;j++)if(mine(cells[j])&&owners[j]!=p)final+=provinceTileValue(cells[j]);
 return final+.25*occupancy/std::max(1,steps);
}

double provinceValue(const Program &q){
 int remaining=q.count-q.cursor;
 if(provinceCount==0||provinceBase>=1000.||remaining<1||remaining>12)return -1e30;
 if(!legalOrder(q.orders[q.cursor])||!permitted(q.orders[q.cursor]))return -1e30;
 double losses[24];
 for(int j=0;j<provinceCount;j++){
  bool feasible=true;losses[j]=provinceOutcome(q,provinceRaids[j],feasible);
  if(!feasible)return -1e30;
 }
 std::sort(losses,losses+provinceCount,std::greater<double>());
 double worst=0;int top=std::min(4,provinceCount);
 for(int j=0;j<top;j++)worst+=losses[j]/top;
 double saved=provinceBase-worst;if(saved<1.)return -1e30;
 // Only reduced losses under these responses pay. A relocation which
 // leaves the same invasion possible receives no concentration prize.
 return 2.*saved/(remaining+1.);
}

void provincePrograms(){
 if(provinceCount==0||provinceBase<3.||provinceBase>=1000.)return;
 // Keep candidate interception posts at several arrival depths. Selecting
 // only the busiest first front squares excludes every feasible rear merge.
 int bands[3][2],counts[3]{},roots[6],count=0;double ranks[3][2]{};
 for(int c=0;c<N;c++)if(mine(c)&&c!=g&&provinceWeight[c]>0){
  int stock=ready(c);
  for(int k=0;k<4;k++){int v=nb[c][k];if(v>=0&&mine(v)){
   bool half=false;if(canDepart(v,half))stock+=moveArmy(v,half);
  }}
  int band=provinceDepth[c]<=4?0:provinceDepth[c]<=8?1:2;
  double value=provinceWeight[c]*std::sqrt(double(stock));
  int j=counts[band];if(j==2){j=1;if(value<=ranks[band][j])continue;}else counts[band]++;
  while(j>0&&value>ranks[band][j-1]){if(j<2){bands[band][j]=bands[band][j-1];ranks[band][j]=ranks[band][j-1];}j--;}
  bands[band][j]=c;ranks[band][j]=value;
 }
 for(int band=2;band>=0;band--)for(int j=0;j<counts[band];j++)roots[count++]=bands[band][j];
 Program best;
 for(int ri=0;ri<count;ri++){
  if(!planningTimeLeft())break;
  int root=roots[ri];scheduleTree(root,10);
  for(int k=1;k<=10;k++)if(schedule[root][k]>ready(root)+3){
   if(!planningTimeLeft())break;
   Program q;q.kind=28;q.root=q.goal=root;q.born=tick;extractSchedule(root,k,q);
   q.value=provinceValue(q);
   if(q.value>best.value)best=q;
  }
 }
 offer(best);
}
