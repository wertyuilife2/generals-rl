// Empire-wide production is not a delivered supply stream. Measure one
// executable owned collection at the current front, using disjoint donors.
// Apply its replacement price only to net depletion of the moving column;
// a transfer through our own one-troop corridor is not a combat casualty.
double frontlineRate=0;
void frontlineRefill(){
 frontlineRate=0;
 if(mainColumn<0)return;
 int enemyMobile=0;bool unknown=false;
 for(int e=0;e<o->players;e++)if(e!=p&&o->alive[e]&&enemyGeneral[e]<0){
  unknown=true;enemyMobile=std::max(enemyMobile,int(std::max<int64_t>(0,o->total_army[e]-o->total_land[e])));
 }
 if(!unknown||ready(mainColumn)>std::max(32,enemyMobile))return;
 double income=.5*(std::max<int64_t>(0,o->total_cities[p])+1)+.02*o->total_land[p];
 double rate=.5;scheduleTree(mainColumn,12);
 for(int budget=1;budget<=12;budget++)if(schedule[mainColumn][budget]>ready(mainColumn)){
  Program q;extractSchedule(mainColumn,budget,q);if(!q.count)continue;
  int added=collectionArrival(q,mainColumn)-ready(mainColumn);
  rate=std::max(rate,double(added)/q.count);
 }
 frontlineRate=std::min(income*2.5+2.,rate);
}
double frontlineReplacementCost(int source,int initial,int remaining,double ordinary)const {
 if(source!=mainColumn||frontlineRate<=0)return 0.;
 return std::max(0,initial-remaining)*std::max(0.,1./frontlineRate-1./ordinary);
}
