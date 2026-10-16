// Whole-position opening search. Each node contains our own armies and
// ownership only. Unknown traversable cells are hypotheses, corrected by
// the next legal observation; no opponent board or action is queried.
struct EconomyState {
 int troops[MAXN];unsigned char owned[MAXN];
 int land=0;double value=0;std::uint64_t hash=0;std::int64_t first=0;
};
struct EconomyChoice {
 double value;std::uint64_t hash;std::int64_t first;
 int parent,s,t,strength;bool half;
};
EconomyState economyStates[2][32];
EconomyChoice economyChoices[4096];
std::uint64_t economyKeys[8192];
int economySlots[8192];
bool economyPass[MAXN],openingContact=false;
int economyGuard[MAXN];

std::uint64_t economyHash(int c,int army)const {
 std::uint64_t x=std::uint64_t(c+1)*0x9e3779b97f4a7c15ULL+army;
 x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;
 return x^(x>>31);
}

int economyFront(const EconomyState &s,int c)const {
 int count=0;for(int k=0;k<4;k++){
  int v=nb[c][k];if(v>=0&&economyPass[v]&&!s.owned[v]&&economyGuard[v]==0)count++;
 }return count;
}

double economyCell(const EconomyState &s,int c)const {
 if(!s.owned[c])return 0;
 int free=economyFront(s,c),mobile=std::max(0,s.troops[c]-1);
 return s.troops[c]+1.6+(st[c]==1?40.:0)+.18*free+
   .28*std::min(mobile,free)-.025*std::min(30,gd[c])+.04*unknown[c];
}

void copyEconomy(EconomyState &to,const EconomyState &from){
 std::memcpy(to.troops,from.troops,N*sizeof(int));
 std::memcpy(to.owned,from.owned,N);to.land=from.land;to.value=from.value;to.hash=from.hash;to.first=from.first;
}

void applyEconomyMove(EconomyState &s,int from,int to,bool half){
 if(from<0)return;
 int sent=half?s.troops[from]/2:s.troops[from]-1;
 s.troops[from]-=sent;
 if(s.owned[to])s.troops[to]+=sent;
 else {s.owned[to]=1;s.land++;s.troops[to]=sent-economyGuard[to];}
}

std::int64_t openingSearch(){
 for(int c=0;c<N;c++)if(o->visible[c]&&own[c]>=0&&own[c]!=p)openingContact=true;
 if(openingContact||tick>=200||baselineDanger>0)return -1;
 auto &initial=economyStates[0][0];initial.value=0;initial.hash=0;initial.first=0;initial.land=0;
 for(int c=0;c<N;c++){
  economyPass[c]=!blocked(c)&&(own[c]<0||mine(c));
  economyGuard[c]=mine(c)?0:rememberedDefense(c);
  initial.owned[c]=mine(c);initial.troops[c]=mine(c)?army(c):0;
  if(initial.owned[c])initial.land++;
  if(initial.owned[c])initial.hash^=economyHash(c,initial.troops[c]);
 }
 for(int c=0;c<N;c++)initial.value+=economyCell(initial,c);
 int current=0,count=1,work=0;
 std::int64_t answer=0;
 double discount=1.;
 for(int depth=1;depth<=24&&work<60000;depth++){
  int generated=0;
  std::fill(economySlots,economySlots+8192,-1);
  auto offerChoice=[&](const EconomyChoice &choice){
   unsigned slot=unsigned(choice.hash)&8191;
   int probes=0;
   while(economySlots[slot]>=0&&economyKeys[slot]!=choice.hash&&++probes<64)slot=(slot+1)&8191;
   if(probes>=64||generated==4096)return;
   if(economySlots[slot]>=0){
    int j=economySlots[slot];if(choice.value>economyChoices[j].value)economyChoices[j]=choice;
   }else{economyKeys[slot]=choice.hash;economySlots[slot]=generated;economyChoices[generated++]=choice;}
  };
  for(int b=0;b<count&&work<60000;b++){
   auto &state=economyStates[current][b];
   int at=actionTurn(depth);
   if(depth>1||growthBeforeMove)for(int c=0;c<N;c++)if(state.owned[c]&&((st[c]==1||st[c]==2)?at%2==0:at%50==0)){
    state.value-=discount*economyCell(state,c);state.hash^=economyHash(c,state.troops[c]);
    state.troops[c]++;state.hash^=economyHash(c,state.troops[c]);state.value+=discount*economyCell(state,c);
   }
   int largest[3]={-1,-1,-1};
   for(int c=0;c<N;c++)if(state.owned[c])for(int j=0;j<3;j++)if(largest[j]<0||state.troops[c]>state.troops[largest[j]]){
    for(int k=2;k>j;k--)largest[k]=largest[k-1];largest[j]=c;break;
   }
   offerChoice({state.value,state.hash,depth==1?0:state.first,b,-1,-1,state.troops[largest[0]],false});
   int roots[16],rootCount=0;double quality[16];
   for(int c=0;c<N;c++)if(state.owned[c]&&state.troops[c]>1){
    int front=economyFront(state,c);
    double q=(front?5.:0)+2.*std::min(50,state.troops[c]-1);
    for(int k=0;k<4;k++){int v=nb[c][k];if(v>=0&&state.owned[v])q+=.04*std::min(30,state.troops[v]-1);}
    if(c==g)q+=1.;
    int j=rootCount;if(j==16){j=15;if(q<=quality[j])continue;}else rootCount++;
    while(j>0&&q>quality[j-1]){if(j<16){roots[j]=roots[j-1];quality[j]=quality[j-1];}j--;}
    roots[j]=c;quality[j]=q;
   }
   for(int ri=0;ri<rootCount&&work<60000;ri++){
    int source=roots[ri],before=state.troops[source];
    for(int k=0;k<4&&work<60000;k++){
     int target=nb[source][k];if(target<0||!economyPass[target])continue;
     for(int half=0;half<2;half++){
      if(half&&before<4)continue;
      int sent=half?before/2:before-1;
      if(before-sent<reserve[source]||(!state.owned[target]&&sent<=economyGuard[target]))continue;
      if(depth==1&&risk(source,target,sent)>=1000)continue;
      work++;
      int affected[6],n=0;
      auto add=[&](int c){if(c<0)return;for(int j=0;j<n;j++)if(affected[j]==c)return;affected[n++]=c;};
      add(source);add(target);for(int j=0;j<4;j++)add(nb[target][j]);
      double value=state.value;
      for(int j=0;j<n;j++)value-=discount*economyCell(state,affected[j]);
      int oldTarget=state.troops[target];bool owned=state.owned[target];
      std::uint64_t hash=state.hash^economyHash(source,before);
      if(owned)hash^=economyHash(target,oldTarget);
      applyEconomyMove(state,source,target,half);
      for(int j=0;j<n;j++)value+=discount*economyCell(state,affected[j]);
      hash^=economyHash(source,state.troops[source])^economyHash(target,state.troops[target]);
      auto first=depth==1?arena_move(source,k,half):state.first;
      int strength=std::max(state.troops[source],state.troops[target]);
      for(int c:largest)if(c>=0&&c!=source&&c!=target){strength=std::max(strength,state.troops[c]);break;}
      offerChoice({value,hash,first,b,source,target,strength,bool(half)});
      state.troops[source]=before;state.troops[target]=oldTarget;state.owned[target]=owned;if(!owned)state.land--;
     }
    }
   }
  }
  if(!generated)break;
  std::sort(economyChoices,economyChoices+generated,
   [](const EconomyChoice &a,const EconomyChoice &b){return a.value>b.value;});
  int selected[32],nextCount=0,bins[8]{};
  bool used[4096]{};
  for(int j=0;j<generated&&nextCount<32;j++){
   int bin=std::min(7,31-__builtin_clz(unsigned(std::max(1,economyChoices[j].strength))));
   if(bins[bin]>=4)continue;
   bins[bin]++;used[j]=true;selected[nextCount++]=j;
  }
  for(int j=0;j<generated&&nextCount<32;j++)if(!used[j])selected[nextCount++]=j;
  // The first retained state is the best score, while the other slots also
  // preserve investment states that have yet to turn troops into territory.
  for(int j=0;j<nextCount;j++){
   auto &choice=economyChoices[selected[j]];auto &next=economyStates[1-current][j];
   copyEconomy(next,economyStates[current][choice.parent]);applyEconomyMove(next,choice.s,choice.t,choice.half);
   next.value=choice.value;next.hash=choice.hash;next.first=choice.first;
  }
  discount*=.97;
  current=1-current;count=nextCount;answer=economyStates[current][0].first;
 }
 return answer;
}
