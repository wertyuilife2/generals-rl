// A short capture and a distant home recall compete for the same army.
// Compare their deadlines before issuing the first step of a repeated
// retreat/advance cycle. This uses visible and remembered material only;
// new observations still revalidate every next order.
bool longRecall=false;
int longRecallEnemy=-1;
bool visibleRaceFeasible(const Program &q)const {
 int goal=q.goal;
 if(goal<0||st[goal]!=2||own[goal]<0||own[goal]==p||q.count-q.cursor<=1)return true;
 // These short tactical programmes already search the opponent's replies.

 if(q.orders[q.count-1].t!=goal)return true;
 int enemy=own[goal],cells[81],owners[81],troops[81],count=0;
 auto index=[&](int c){for(int i=0;i<count;i++)if(cells[i]==c)return i;return -1;};
 for(int j=q.cursor;j<q.count;j++)for(int c:{q.orders[j].s,q.orders[j].t}){
  if(c<0||c>=N||blocked(c))return false;
  if(index(c)>=0)continue;
  cells[count]=c;owners[count]=own[c];troops[count]=mine(c)?ready(c):rememberedDefense(c);count++;
 }
 int king=index(goal),arrival=0;
 for(int j=q.cursor;j<q.count;j++){
  int step=j-q.cursor+1;
  if(step>1)for(int i=0;i<count;i++)if(owners[i]>=0)troops[i]+=grows(cells[i],actionTurn(step));
  auto x=q.orders[j];int s=index(x.s),t=index(x.t);
  if(owners[s]!=p)return false;
  int sent=x.half?troops[s]/2:troops[s]-1;if(sent<1)return false;
  troops[s]-=sent;
  if(owners[t]==p)troops[t]+=sent;
  else if(sent>troops[t]){troops[t]=sent-troops[t];owners[t]=p;}
  else troops[t]-=sent;
  if(owners[king]==p){arrival=step;break;}
 }
 if(!arrival)return false;
 if(arrival==1)return true;
 // Give the defender every visible off-route army that can reach a
 // source or landing by that order, ignoring movement losses and shared
 // orders. This overestimates simultaneous visible replies, and never
 // charges the same army once for interception and again for reinforcement.
 // Hidden reinforcements remain uncertainty, not invented map contents.
 long long reply=0;
 for(int source=0;source<N;source++)if(source!=goal&&o->visible[source]&&own[source]==enemy){
  int at=index(source);if(at>=0&&owners[at]==p)continue;
  bool reachable=false;
  for(int j=q.cursor;j<q.cursor+arrival&&!reachable;j++){
   int steps=j-q.cursor+1;
   for(int target:{q.orders[j].s,q.orders[j].t})
    if(std::abs(cellX[source]-cellX[target])+std::abs(cellY[source]-cellY[target])<=steps)reachable=true;
  }
  if(reachable)reply+=std::max(0,ready(source)-1)+futureGrowth(arrival,(st[source]>0&&st[source]<3)?2:50);
 }
 return troops[king]>reply;
}

bool outrunsLongRecall(const Program &q)const {
 if(!longRecall||aliveCount!=2||baselineDanger>0||q.cursor>=q.count)return false;
 int goal=q.goal,steps=q.count-q.cursor;
 if(goal<0||st[goal]!=2||own[goal]!=longRecallEnemy||steps>6)return false;
 if(q.orders[q.count-1].t!=goal)return false;
 if(!permitted(q.orders[q.cursor]))return false;
 return visibleRaceFeasible(q);
}
