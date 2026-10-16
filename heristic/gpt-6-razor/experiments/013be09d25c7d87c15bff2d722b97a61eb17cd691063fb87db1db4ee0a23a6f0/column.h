// A half scout and a whole-column raid spend the same strategic resource.
// Compare the executable terminal army for every non-tactical programme,
// rather than forbidding only plans labelled reconnaissance. Collection,
// friendly relocation and a split which joins a second army keep their force.
double columnCapitalRetention(const Program &q)const {
 if(mainColumn<0||mainColumnNeed<=0||q.cursor>=q.count)return 1.;
 if(q.kind==3||q.kind==4||q.kind==9||q.kind==14||q.kind==17||q.kind==20||q.kind==21)return 1.;
 if(q.goal>=0&&st[q.goal]==2&&own[q.goal]>=0&&own[q.goal]!=p)return 1.;
 bool spending=false;
 for(int j=q.cursor;j<q.count;j++)if(q.orders[j].s==mainColumn){spending=true;break;}
 if(!spending)return 1.;
 int index[MAXN],cells[81],owners[81],troops[81],count=0;
 std::fill(index,index+N,-1);
 for(int j=q.cursor;j<q.count;j++)for(int c:{q.orders[j].s,q.orders[j].t}){
  if(c<0||c>=N||blocked(c))return .25;
  if(index[c]>=0)continue;
  int i=count++;index[c]=i;cells[i]=c;owners[i]=own[c];
  troops[i]=mine(c)?ready(c):rememberedDefense(c);
 }
 for(int j=q.cursor;j<q.count;j++){
  int step=j-q.cursor+1;
  if(step>1)for(int i=0;i<count;i++)if(owners[i]>=0)troops[i]+=grows(cells[i],actionTurn(step));
  auto x=q.orders[j];int a=index[x.s],b=index[x.t];
  if(owners[a]!=p)return .25;
  int sent=x.half?troops[a]/2:troops[a]-1;
  if(sent<1||troops[a]-sent<reserve[x.s])return .25;
  troops[a]-=sent;
  if(owners[b]==p)troops[b]+=sent;
  else {if(sent<=troops[b])return .25;troops[b]=sent-troops[b];owners[b]=p;}
 }
 int retained=0;
 for(int i=0;i<count;i++)if(owners[i]==p){
  int available=troops[i]-1;
  if(reserve[cells[i]]>1)available=troops[i]/2;
  if(troops[i]-available>=reserve[cells[i]])retained=std::max(retained,available);
 }
 // An existing independent field column is still available. A donor spent
 // by this programme is represented by the final ledger above, not twice.
 for(int c=0;c<N;c++)if(index[c]<0&&mine(c)&&st[c]==0&&ed[c]<=10){
  bool half=false;if(canDepart(c,half))retained=std::max(retained,moveArmy(c,half));
 }
 int before=std::min(mainColumnNeed,ready(mainColumn)-1);
 int depleted=std::max(0,before-retained);if(!depleted)return 1.;
 double income=.5*(std::max<int64_t>(0,o->total_cities[p])+1)+.02*o->total_land[p];
 double steps=q.count-q.cursor+1.;
 return steps/(steps+depleted/(income*2.5+2.));
}
