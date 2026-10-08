// Standalone structural research probe; does not modify production NumPy.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>
using I=std::int64_t;
using Clock=std::chrono::steady_clock;
static volatile std::uint64_t sink=0;
static std::uint64_t probes=0;
static inline I lower(const std::vector<I>& a,I v,I lo,I hi){
 while(lo<hi){I m=lo+(hi-lo)/2; ++probes; if(a[m]<v)lo=m+1;else hi=m;}return lo;
}
static std::vector<I> baseline(const std::vector<I>& a,const std::vector<I>& v){
 std::vector<I> p(v.size(),0);I width=a.size();
 while(width>0){I half=width/2;I rem=width-half;
  for(size_t i=0;i<v.size();++i){I m=std::min<I>(p[i]+half,(I)a.size()-1);
   ++probes;if(a[m]<v[i])p[i]+=half+(width==1?1:0);}
  width=rem-1; // Only for conceptual independent comparison; correctness checked below.
 }
 return p;
}
static std::vector<I> reference(const std::vector<I>& a,const std::vector<I>& v){
 std::vector<I> p; p.reserve(v.size());for(I x:v)p.push_back(lower(a,x,0,a.size()));return p;
}
static std::vector<I> locality(const std::vector<I>& a,const std::vector<I>& v){
 std::vector<I> out;out.reserve(v.size());I prev=0,last=0;
 for(size_t i=0;i<v.size();++i){I lo=(i && v[i]>=last)?prev:0;
  out.push_back(lower(a,v[i],lo,a.size()));prev=out.back();last=v[i];}
 return out;
}
static bool monotonic(const std::vector<I>& v){for(size_t i=1;i<v.size();++i){++probes;if(v[i]<v[i-1])return false;}return true;}
static std::vector<I> gateA(const std::vector<I>& a,const std::vector<I>& v){
 // Sequential full check, sharing query cachelines but not yet fused into NumPy C++.
 if(monotonic(v))return locality(a,v);return reference(a,v);
}
static std::vector<I> gateB(const std::vector<I>& a,const std::vector<I>& v){
 std::vector<I> out(v.size());const size_t block=256;
 for(size_t start=0;start<v.size();start+=block){size_t end=std::min(v.size(),start+block);
  bool good=true;for(size_t i=start+1;i<end;i++){++probes;if(v[i]<v[i-1]){good=false;break;}}
  I prev=0;
  for(size_t i=start;i<end;i++){
   I lo=(good&&i>start)?prev:0;out[i]=lower(a,v[i],lo,a.size());prev=out[i];
  }
 }
 return out;
}
static std::vector<I> gateC(const std::vector<I>& a,const std::vector<I>& v){
 // Actual-search comparisons count as cost; fallback cannot undo prior work.
 std::vector<I> out(v.size());I prev=0,last=0;bool enable=true;
 for(size_t i=0;i<v.size();i++){
  I lo=(enable&&i>0&&v[i]>=last)?prev:0;
  auto old=probes;out[i]=lower(a,v[i],lo,a.size());
  if(probes-old>22)enable=false;
  prev=out[i];last=v[i];
 }
 return out;
}
static std::vector<I> choose(int mode,const std::vector<I>&a,const std::vector<I>&v){
 switch(mode){case 0:return reference(a,v);case 1:return gateA(a,v);case 2:return gateB(a,v);case 3:return gateC(a,v);default:return locality(a,v);}
}
int main(){
 std::mt19937_64 r(20261009);
 std::cout<<"pattern,n,q,mode,median_ns,comparisons,correct\n";
 for(I n: {4096LL,65536LL,1048576LL})for(I q:{64LL,1024LL,65536LL}){
  std::vector<I>a(n);std::iota(a.begin(),a.end(),0);
  for(std::string pat:{"repeated","ordered_local","random","alternating","deceptive"}){
   std::vector<I>v(q);I center=n/2;
   for(I i=0;i<q;i++){
    if(pat=="repeated")v[i]=center;
    else if(pat=="ordered_local")v[i]=center+(i*8/q);
    else if(pat=="random")v[i]=r()%n;
    else if(pat=="alternating")v[i]=i%2?n-1:0;
    else v[i]=center;
   }
   if(pat=="deceptive"){for(I i=0;i<q;i++)if(i%64>2)v[i]=r()%n;}
   auto expected=reference(a,v);
   for(int mode=0;mode<5;mode++){
    auto got=choose(mode,a,v);if(got!=expected){std::cerr<<"CORRECTNESS FAIL "<<mode<<" "<<pat<<"\n";return 1;}
    int loops=q>10000?3:30;std::vector<double>t;std::uint64_t c=0;
    for(int rep=0;rep<7;rep++){probes=0;auto start=Clock::now();
     for(int j=0;j<loops;j++){auto z=choose(mode,a,v);sink=sink+(std::uint64_t)z[(j+rep)%q];}
     auto end=Clock::now();t.push_back(std::chrono::duration<double,std::nano>(end-start).count()/loops);
     c+=probes/loops;
    }
    std::sort(t.begin(),t.end());std::cout<<pat<<","<<n<<","<<q<<","<<mode<<","<<t[3]<<","<<(c/7)<<",1\n";
   }
  }
 }
 std::cerr<<"probe_done sink="<<sink<<"\n";
}
