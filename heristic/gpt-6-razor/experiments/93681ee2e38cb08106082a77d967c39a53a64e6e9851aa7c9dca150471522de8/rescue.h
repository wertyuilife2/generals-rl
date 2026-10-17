// A short king rescue must survive the same approach paths that produced the
// alarm. A shared sparse ledger prevents spent, captured or cut-off donors
// from also being delivered to home. This is a bounded visible-path model.
bool rescueRoute(const Program &q,const AttackRoute *route,const int *adjacent,int adjacentCount)const {
 int cells[42],owners[42],troops[42],count=0;
 auto index=[&](int c){for(int j=0;j<count;j++)if(cells[j]==c)return j;
  int j=count++;cells[j]=c;owners[j]=own[c];troops[j]=mine(c)?ready(c):rememberedDefense(c);return j;};
 int home=index(g),enemy=-1;
 for(int e=0;e<o->players;e++)if(e!=p&&o->alive[e])enemy=e;
 if(enemy<0)return false;
 if(route)for(int j=0;j<=route->length;j++)index(route->cell[j]);
 else for(int j=0;j<adjacentCount;j++)index(adjacent[j]);
 for(int j=q.cursor;j<q.count;j++){index(q.orders[j].s);index(q.orders[j].t);}
 int position=0;bool stopped=false;
 auto move=[&](int source,int target,bool half,int player){
  int s=index(source),t=index(target);if(owners[s]!=player)return false;
  int sent=half?troops[s]/2:troops[s]-1;if(sent<=0)return false;
  troops[s]-=sent;
  if(owners[t]==player)troops[t]+=sent;
  else if(sent>troops[t]){troops[t]=sent-troops[t];owners[t]=player;}
  else troops[t]-=sent;
  return true;
 };
 int horizon=route?route->length:adjacentCount;
 for(int step=0;step<horizon;step++){
  if(step)for(int j=0;j<count;j++)if(owners[j]>=0)troops[j]+=grows(cells[j],actionTurn(step+1));
  auto attack=[&](){
   if(route){
    if(stopped||position>=route->length)return;
    int s=route->cell[position],t=route->cell[position+1];
    if(!move(s,t,false,enemy)||owners[index(t)]!=enemy){stopped=true;return;}
    position++;
   }else if(step<adjacentCount)move(adjacent[step],g,false,enemy);
  };
  if(enemy<p)attack();
  if(owners[home]!=p)return false;
  if(q.cursor+step<q.count){
   auto x=q.orders[q.cursor+step];
   if(!move(x.s,x.t,x.half,p))return false;
  }
  if(enemy>p)attack();
  if(owners[home]!=p)return false;
 }
 return true;
}

bool rescueHolds(const Program &q)const {
 if(aliveCount!=2||q.count-q.cursor<1||q.count-q.cursor>8)return false;
 int enemy=-1;for(int e=0;e<o->players;e++)if(e!=p&&o->alive[e])enemy=e;
 if(enemy<0)return false;
 for(int d=1;d<=8;d++)for(int j=0;j<attackCounts[enemy][d];j++)
  if(!rescueRoute(q,&attacks[enemy][d][j],nullptr,0))return false;
 int adjacent[4],count=0;
 for(int k=0;k<4;k++){int c=nb[g][k];if(c>=0&&o->visible[c]&&own[c]==enemy&&moveArmy(c)>0)adjacent[count++]=c;}
 // Each adjacent army can attack on a different turn. Test both directions
 // of strength order rather than treating every army as an isolated raid.
 if(count>1){
  std::sort(adjacent,adjacent+count,[&](int a,int b){return ready(a)>ready(b);});
  if(!rescueRoute(q,nullptr,adjacent,count))return false;
  std::reverse(adjacent,adjacent+count);
  if(!rescueRoute(q,nullptr,adjacent,count))return false;
 }
 return true;
}

void rescuePrograms(){
 if(aliveCount!=2||baselineDanger<=0)return;
 scheduleTree(g,8);
 for(int k=1;k<=8;k++){
  if(schedule[g][k]<ready(g)+1)continue;
  Program q;q.root=q.goal=g;q.born=tick;q.kind=3;q.rescueModel=1;
  extractSchedule(g,k,q);
  if(!q.count||!legalOrder(q.orders[0],true)||!rescueHolds(q))continue;
  // A feasible complete rescue has priority over an unvalidated partial
  // transfer. Immediate enemy-general capture keeps its terminal priority.
  q.value=baselineDanger+40.;offer(q);return;
 }
}
