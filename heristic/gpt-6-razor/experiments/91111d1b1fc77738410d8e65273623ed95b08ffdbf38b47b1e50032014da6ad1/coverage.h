// A field column is also an option to defend a production asset. Spending
// that option must be priced before the move, not after the column has left
// the owned return route. These are bounded single-column estimates, not a
// claim that several assets can all use the same troops simultaneously.
struct CoverAsset {
 int root=-1,need=0,budget=0;
 int donor[3]={-1,-1,-1},credit[3]{};
 double price=0;
};
CoverAsset coverAssets[7];
short coverDistance[7][MAXN];
int coverCount=0;

int homeInsurance()const {
 if(tick<70||raidValue<=0)return 0;
 int need=0;
 for(int e=0;e<o->players;e++)if(homeKnownTo[e]&&o->alive[e]){
  int pulses=(tick+6)/50-tick/50;
  int field=int(std::min<int64_t>(300,std::max<int64_t>(0,
       o->total_army[e]-o->total_land[e]+pulses*o->total_land[e])));
  need=std::max(need,int(raidValue*std::min(field*.40,double(o->total_army[p])*.30)));
 }
 return need;
}

int returnCredit(int asset,int cell,int troops,int distance)const {
 const auto &c=coverAssets[asset];
 if(cell==c.root||distance<1||distance>c.budget)return 0;
 int sent=reserve[cell]>1?troops/2:troops-1;
 if(troops-sent<reserve[cell])return 0;
 return std::max(0,sent-distance+1);
}

void buildCoverage(){
 coverCount=0;
 int insurance=homeInsurance();
 if(insurance>0)coverAssets[coverCount++]={g,insurance,6,{-1,-1,-1},{},.6};
 // There is at most one order per seat. Keep the most important threatened
 // assets rather than summing a separate imaginary mobilisation at each city.
 for(int c=0;c<N;c++)if(mine(c)&&st[c]==1&&zoneETA[c]<=6&&zoneThreat[c]>4){
  int need=zoneThreat[c]-futureGrowth(zoneETA[c])+2;
  int budget=std::max(0,zoneETA[c]-1);
  if(need>std::max<int64_t>(40,o->total_army[p]))continue;
  CoverAsset asset{c,std::max(1,need),budget,{-1,-1,-1},{},.22};
  if(coverCount<7)coverAssets[coverCount++]=asset;
  else{
   int worst=-1;
   for(int j=0;j<coverCount;j++)if(coverAssets[j].root!=g&&
       (worst<0||coverAssets[j].need<coverAssets[worst].need))worst=j;
   if(worst>=0&&need>coverAssets[worst].need)coverAssets[worst]=asset;
  }
 }
 for(int j=0;j<coverCount;j++){
  auto &c=coverAssets[j];auto *distance=coverDistance[j];
  std::fill(distance,distance+N,short(-1));
  int head=0,tail=1;queue[0]=c.root;distance[c.root]=0;
  while(head<tail){
   int u=queue[head++];
   if(distance[u]>=c.budget||(u!=c.root&&reserve[u]>1))continue;
   for(int k=0;k<4;k++){
    int v=nb[u][k];if(v<0||!mine(v)||distance[v]>=0)continue;
    distance[v]=distance[u]+1;queue[tail++]=v;
   }
  }
  for(int k=1;k<tail;k++){
   int u=queue[k],value=returnCredit(j,u,ready(u),distance[u]);
   if(value<=c.credit[2])continue;
   int pos=2;while(pos>0&&value>c.credit[pos-1]){
    c.credit[pos]=c.credit[pos-1];c.donor[pos]=c.donor[pos-1];pos--;
   }
   c.credit[pos]=value;c.donor[pos]=u;
  }
 }
}

double coverageCost(const Order &x)const {
 if(!coverCount)return 0.;
 int moving=moveArmy(x.s,x.half),left=ready(x.s)-moving;
 int arriving=moving+(mine(x.t)?ready(x.t):-rememberedDefense(x.t));
 if(arriving<1)return 0.;
 double worst=0.;
 for(int j=0;j<coverCount;j++){
  const auto &c=coverAssets[j];
  // Removing the threatening column is itself a military use of the cover.
  if(o->visible[x.t]&&own[x.t]>=0&&own[x.t]!=p&&army(x.t)*2>=c.need)continue;
  int before=ready(c.root)+c.credit[0],best=0;
  for(int k=0;k<3;k++)if(c.donor[k]!=x.s&&c.donor[k]!=x.t)best=std::max(best,c.credit[k]);
  best=std::max(best,returnCredit(j,x.s,left,coverDistance[j][x.s]));
  int d=coverDistance[j][x.t];
  if(!mine(x.t)){
   d=MAXN;
   for(int k=0;k<4;k++){
    int v=nb[x.t][k];if(v<0||!mine(v)||coverDistance[j][v]<0)continue;
    if(v!=c.root&&reserve[v]>1)continue;
    d=std::min(d,1+int(coverDistance[j][v]));
   }
  }
  best=std::max(best,returnCredit(j,x.t,arriving,d));
  int held=c.root==x.s?left:c.root==x.t?arriving:ready(c.root);
  int lost=std::min(c.need,before)-std::min(c.need,held+best);
  // Use the largest new deficit; do not charge several cities for losing
  // access to the same army. Existing deficits do not forbid useful play.
  worst=std::max(worst,c.price*lost);
 }
 return worst;
}
