// A field army keeps the capital invested in its current sortie. Updating
// the one-step route cannot erase earlier depletion. All history is private
// to this seat and all transfers must be confirmed by legal observations.
int campaignPeak[MAXN]{},campaignOldPeak[MAXN]{};
int campaignOldArmy[MAXN]{},campaignOldOwner[MAXN]{};
int campaignTick=-1;

void campaignTransfer(int s,int t,int moved,bool half){
 if(s<0||s>=N||t<0||t>=N||!mine(t)||moved<=0)return;
 int source=campaignOldPeak[s],target=campaignOldPeak[t];
 if(campaignOldOwner[t]!=p)target=0;
 if(half){
  // The departing scout gets its own actual initial capital. It cannot
  // renew the allowance of the stationary parent column every turn.
  int arriving=moved+(campaignOldOwner[t]==p?campaignOldArmy[t]:0);
  campaignPeak[t]=std::max({target,arriving,army(t)});
 }else{
  campaignPeak[t]=std::max({source,target,army(t)});
  if(mine(s))campaignPeak[s]=army(s)>=24?army(s):0;
 }
}

void campaignLedger(){
 std::copy(campaignPeak,campaignPeak+N,campaignOldPeak);
 for(int c=0;c<N;c++){
  if(!mine(c)){campaignPeak[c]=0;continue;}
  // A general's accumulating home garrison is not an already deployed
  // sortie. Its outgoing army starts at the actually delivered strength.
  if(c==g){campaignPeak[c]=0;continue;}
  if(army(c)>=24||campaignPeak[c]>0)campaignPeak[c]=std::max(campaignPeak[c],army(c));
 }
 bool transferred=false;
 if(campaignTick>=0&&tick>campaignTick&&o->previous&&o->previous->valid){
  const auto &r=*o->previous;
  if(r.source>=0&&r.source<N&&r.target>=0&&r.target<N&&campaignOldOwner[r.source]==p){
   campaignTransfer(int(r.source),int(r.target),int(r.moved),r.action>0&&bool((r.action-1)%2));transferred=true;
  }
 }
 if(!transferred&&campaignTick==tick-1&&!o->previous){
  // Online observations may lack receipts. Require a matching source loss
  // AND adjacent destination gain, not merely a large stack at a new cell.
  // Ambiguous exchanges or skipped observations do not fabricate a move.
  for(int s=0;s<N&&!transferred;s++)if(campaignOldOwner[s]==p&&campaignOldArmy[s]>=16&&mine(s)){
   int moved=campaignOldArmy[s]+grows(s,tick)-army(s);if(moved<8)continue;
   bool half=std::abs(moved-campaignOldArmy[s]/2)<=1;
   bool full=army(s)<=2;if(!half&&!full)continue;
   for(int k=0;k<4;k++){
    int t=nb[s][k];if(t<0||!mine(t))continue;
    int old=campaignOldArmy[t];
    if(campaignOldOwner[t]==p&&army(t)<old+8)continue;
    int expected=moved+(campaignOldOwner[t]==p?old:-old);
    if(expected<1||std::abs(army(t)-expected)>2)continue;
    campaignTransfer(s,t,moved,half&&!full);transferred=true;break;
   }
  }
 }
 for(int c=0;c<N;c++){if(army(c)<=2)campaignPeak[c]=0;campaignOldArmy[c]=army(c);campaignOldOwner[c]=o->owner[c];}
 campaignTick=tick;
}

double campaignRaidRent(int source,int target,int remaining)const {
 if(source<0||source>=N||campaignPeak[source]<32||own[target]<0||own[target]==p||st[target]!=0)return 1.;
 int enemy=own[target];if(enemyGeneral[enemy]>=0||!o->alive[enemy])return 1.;
 double peak=campaignPeak[source];
 double potential=std::max<int64_t>(0,o->total_army[enemy]-o->total_land[enemy]);
 double cover=std::min(.75*peak,std::max(16.,potential*std::max(.35,observedColumnFraction[enemy])));
 // Spend the operational margin on ordinary raiding. Information and
 // concrete king attacks retain separate rewards when that margin is gone.
 return std::clamp((remaining-cover)/std::max(1.,peak-cover),0.,1.);
}
