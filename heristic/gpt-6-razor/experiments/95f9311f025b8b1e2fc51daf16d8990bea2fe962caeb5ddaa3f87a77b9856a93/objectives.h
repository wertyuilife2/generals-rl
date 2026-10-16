// All generators share the same terminal test. A collection, transport or
// occupation label cannot turn an unfunded general capture into a different
// source of elimination reward.
bool kingProgramFeasible(const Program &q)const {
 int goal=q.goal;
 if(goal<0||st[goal]!=2||own[goal]<0||own[goal]==p||q.count-q.cursor<=1)return true;
 // These short tactical programmes already search the opponent's replies.
 if(q.kind==17||q.kind==21)return true;
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
 int charged=0;
 for(int i=0;i<count;i++)if(cells[i]!=goal&&owners[i]==p&&own[cells[i]]==enemy&&o->visible[cells[i]])
  charged+=std::max(0,ready(cells[i])-1);
 int intercept=0;
 for(int j=q.cursor;j<q.cursor+arrival;j++)for(int k=0;k<4;k++){
  int source=nb[q.orders[j].t][k];
  if(source<0||source==goal||!o->visible[source]||own[source]!=enemy)continue;
  int at=index(source);if(at>=0&&owners[at]==p)continue;
  // One observed off-route army may wait and then hit the approaching
  // column. It need not defeat that column to remove its capture margin,
  // and it need not have an enemy-owned road back to its general.
  intercept=std::max(intercept,moveArmy(source));
 }
 // Compare alternative replies rather than counting the same donor twice.
 int reply=std::max(intercept,std::max(0,kingRelief[enemy][std::min(32,arrival)]-charged));
 return troops[king]>reply;
}
