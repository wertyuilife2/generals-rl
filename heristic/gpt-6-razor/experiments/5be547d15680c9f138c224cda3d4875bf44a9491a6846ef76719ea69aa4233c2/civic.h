// A controlled city includes a returning field force, not just its displayed
// garrison. Credit one concrete friendly return path; separate donors require
// a real collection program. These visible-force estimates rank opportunities
// and do not claim certainty about unseen enemy reinforcements.
int civicCount=0,civicCity[2],civicEnemy[2],civicETA[2],civicNeed[2],civicBefore[2];
int civicDistance[2][MAXN],civicNext[2][MAXN];

int civicCover(int slot,const int *index=nullptr,const int *owners=nullptr,
               const int *troops=nullptr,int elapsed=0)const {
 auto ownerAt=[&](int c){return index&&index[c]>=0?owners[index[c]]:own[c];};
 auto forceAt=[&](int c){
  if(index&&index[c]>=0)return troops[index[c]];
  int period=st[c]>0&&st[c]<3?2:50;
  return ready(c)+std::max(0,actionTurn(elapsed+1)/period-actionTurn(1)/period);
 };
 int root=civicCity[slot];if(ownerAt(root)!=p)return 0;
 int best=forceAt(root),budget=std::max(0,civicETA[slot]-1);
 for(int source=0;source<N;source++){
  int distance=civicDistance[slot][source];
  if(source==root||distance<1||distance>budget||ownerAt(source)!=p)continue;
  int force=forceAt(source),u=source;bool valid=true;
  for(int step=1;step<=distance;step++){
   int v=civicNext[slot][u];if(v<0||ownerAt(v)!=p){valid=false;break;}
   int moving=reserve[u]>1?force/2:force-1;
   if(moving<1||force-moving<reserve[u]){valid=false;break;}
   force=moving+forceAt(v);u=v;
  }
  if(valid)best=std::max(best,force);
 }
 return best;
}

void buildCityControl(){
 civicCount=0;if(tick<120)return;
 int cities[2]={-1,-1};double priorities[2]={-1,-1};
 for(int c=0;c<N;c++)if(mine(c)&&st[c]==1&&zoneETA[c]<=6&&zoneThreat[c]>=12){
  double priority=(1.+.2*assetFlips[c])*(12.+std::min(60,zoneThreat[c]))/(2.+zoneETA[c]);
  int j=priority>priorities[0]?0:priority>priorities[1]?1:-1;if(j<0)continue;
  if(j==0){cities[1]=cities[0];priorities[1]=priorities[0];}
  cities[j]=c;priorities[j]=priority;
 }
 for(int root:cities){
  if(root<0)continue;
  bfs(root,dist);int enemy=-1,need=0,eta=INF;
  for(int c=0;c<N;c++)if(o->visible[c]&&own[c]>=0&&own[c]!=p&&o->alive[own[c]]&&dist[c]<=6){
   int arriving=std::max(0,ready(c)-dist[c]);
   if(arriving>need||(arriving==need&&dist[c]<eta)){enemy=c;need=arriving;eta=dist[c];}
  }
  if(enemy<0||need<12)continue;
  int slot=civicCount++;civicCity[slot]=root;civicEnemy[slot]=enemy;
  civicETA[slot]=eta;civicNeed[slot]=need;
  auto d=civicDistance[slot],next=civicNext[slot];
  std::fill(d,d+N,INF);std::fill(next,next+N,-1);
  int head=0,tail=1;queue[0]=root;d[root]=0;
  while(head<tail){int u=queue[head++];if(d[u]>=6)continue;
   for(int k=0;k<4;k++){int v=nb[u][k];if(v<0||!mine(v)||d[v]<INF)continue;
    d[v]=d[u]+1;next[v]=u;queue[tail++]=v;
   }
  }
  civicBefore[slot]=civicCover(slot);
 }
}

// Assess the complete remaining program. Moving one square outside a city
// need not relinquish it: a sufficiently strong returnable army still covers
// the asset. The same troops cannot be counted as two independent donors.
double cityControlCost(const Program &q)const {
 if(!civicCount||q.kind==3||q.kind==4||q.kind==9||q.kind==17||q.kind==20||q.kind==21)return 0.;
 if(q.goal>=0&&st[q.goal]==2&&own[q.goal]>=0&&own[q.goal]!=p)return 0.;
 int index[MAXN],cells[81],owners[81],troops[81],count=0;
 std::fill(index,index+N,-1);
 for(int j=q.cursor;j<q.count;j++)for(int c:{q.orders[j].s,q.orders[j].t}){
  if(c<0||c>=N)return 0.;if(index[c]>=0)continue;
  int i=count++;index[c]=i;cells[i]=c;owners[i]=own[c];
  troops[i]=mine(c)?ready(c):rememberedDefense(c);
 }
 for(int j=q.cursor;j<q.count;j++){
  int step=j-q.cursor+1;
  if(step>1)for(int i=0;i<count;i++)if(owners[i]>=0)troops[i]+=grows(cells[i],actionTurn(step));
  auto x=q.orders[j];int s=index[x.s],t=index[x.t];if(owners[s]!=p)return 0.;
  int moving=x.half?troops[s]/2:troops[s]-1;if(moving<1)return 0.;troops[s]-=moving;
  if(owners[t]==p)troops[t]+=moving;
  else if(moving>troops[t]){troops[t]=moving-troops[t];owners[t]=p;}
  else return 0.;
 }
 int steps=q.count-q.cursor;double cost=0.;
 for(int i=0;i<civicCount;i++){
  int enemy=civicEnemy[i],e=index[enemy];
  if(e>=0&&owners[e]!=own[enemy])continue;
  int threat=civicNeed[i];if(e>=0)threat=std::max(0,threat+troops[e]-ready(enemy));
  int before=civicBefore[i];if(before<threat)continue;
  int after=civicCover(i,index,owners,troops,steps-1);
  if(after>=threat)continue;
  // Forty ticks of potential production, plus opponent denial in a duel.
  double value=20.*(1.+1./std::max(1,aliveCount-1));
  cost+=value*std::min(1.,double(threat-after)/(threat*.5+1.))/(steps+1.);
 }
 return cost;
}

bool civicReliefNeeded(int city)const {
 for(int i=0;i<civicCount;i++)if(civicCity[i]==city)return civicBefore[i]<civicNeed[i];
 return false;
}

void cityReliefPrograms(){
 if(baselineDanger>0)return;
 for(int i=0;i<civicCount;i++){
  if(!planningTimeLeft())break;
  int root=civicCity[i],budget=std::min(6,civicETA[i]-1);
  if(budget<1||civicBefore[i]>=civicNeed[i])continue;
  scheduleTree(root,budget);Program best;
  for(int k=1;k<=budget;k++)if(schedule[root][k]>0){
   Program q;q.kind=26;q.root=q.goal=root;q.born=tick;extractSchedule(root,k,q);
   int force=collectionArrival(q,root);if(force<civicNeed[i]+1)continue;
   double value=20.*(1.+1./std::max(1,aliveCount-1));
   q.value=value/(q.count+1.);
   if(q.count&&permitted(q.orders[0])&&q.value>best.value)best=q;
  }
  offer(best);
 }
}
