#include "../experiments/f77ab97dd53ef9427c1a154fb8c0576026a56fb8106e6739b56e21e4d0ad59a4/policy.cpp"

#include <cassert>
#include <memory>
#include <iostream>
int main(){
 constexpr int N=121;
 uint8_t terrain[N],structure[N]{},visible[N],alive[2]={1,1};
 int16_t owner[N];int64_t army[N]{},total[2]={60,100},land[2]={11,20},cities[2]={1,-1};
 std::fill(terrain,terrain+N,1);std::fill(visible,visible+N,1);std::fill(owner,owner+N,-1);
 for(int y=0;y<11;y++){int c=y*11;terrain[c]=0;owner[c]=0;army[c]=1;}
 for(int c:{1,2,12}){terrain[c]=0;owner[c]=0;army[c]=1;}
 structure[2]=1;army[2]=2;structure[0]=2;army[0]=30;owner[110]=1;army[110]=40;
 ArenaObservation obs{11,11,2,0,134,0,terrain,structure,owner,army,visible,alive,total,land,cities,nullptr};
 auto p=std::make_unique<Policy>(0,1,"{}");p->update(&obs);p->assetHistory();p->pressureMap();p->buildAttacks();
 p->baselineDanger=p->dangerAfter(nullptr,p->homeIncoming);p->prepareProvinceRaids();
 assert(p->baselineDanger==0);assert(p->provinceBase<1000);
 bool retained=false,exposes=false;
 Policy::Program wait,loan;loan.kind=28;loan.root=loan.goal=1;loan.count=1;loan.orders[0]={0,1,true};
 assert(p->permitted(loan.orders[0])); // Outside the eight-order tactical horizon.
 for(int j=0;j<p->provinceCount;j++){
  const auto &r=p->provinceRaids[j];if(r.cell[r.count]!=0)continue;
  retained=true;bool feasible=true;double before=p->provinceOutcome(wait,r,feasible);assert(feasible&&before<1000);
  double after=p->provinceOutcome(loan,r,feasible);assert(feasible);exposes|=after>=1000;
 }
 std::cout<<"retained="<<retained<<" exposes="<<exposes<<" value="<<p->provinceValue(loan)<<std::endl;
 assert(retained&&exposes);assert(p->provinceValue(loan)<0);
}
