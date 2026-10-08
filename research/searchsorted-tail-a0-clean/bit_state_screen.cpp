#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>
using i64=std::int64_t;
static volatile i64 sink=0;
template<int MODE> void search(const std::vector<i64>&a,const std::vector<i64>&keys,std::vector<i64>&out,bool right){
 const i64 n=a.size(),q=keys.size();if(!n){std::fill(out.begin(),out.end(),0);return;}
 if constexpr(MODE==2){
  // One-bit orientation: key >= previous key. Previous exact rank supplies a safe bound.
  i64 previous=0;
  for(i64 i=0;i<q;i++){
   i64 lo=0,hi=n;
   if(i){bool nondecreasing=keys[i]>=keys[i-1];if(nondecreasing)lo=previous;else hi=previous;}
   while(lo<hi){i64 mid=lo+(hi-lo)/2;bool below=right?a[mid]<=keys[i]:a[mid]<keys[i];if(below)lo=mid+1;else hi=mid;}
   out[i]=lo;previous=lo;
  }return;
 }
 i64 length=n,half=length>>1;length-=half;
 for(i64 i=0;i<q;i++)out[i]=(right?a[half]<=keys[i]:a[half]<keys[i])*half;
 while(length>(MODE==1?2:1)){
  half=length>>1;length-=half;
  for(i64 i=0;i<q;i++){i64 &b=out[i];bool below=right?a[b+half]<=keys[i]:a[b+half]<keys[i];b+=below*half;}
 }
 for(i64 i=0;i<q;i++){
  i64 &b=out[i];const i64 k=keys[i];
  if constexpr(MODE==1){if(length>1)b+=(right?a[b+1]<=k:a[b+1]<k);}
  b+=(right?a[b]<=k:a[b]<k);
 }
}
int main(){
 std::mt19937_64 gen(32895);
 int cases=0;
 for(int n=1;n<=129;n++)for(int q: {1,2,15,128}){
  std::vector<i64>a(n),keys(q),o(q),check(q);
  for(auto&v:a)v=gen()%27-13;std::sort(a.begin(),a.end());
  for(auto&v:keys)v=gen()%39-19;
  for(bool right:{false,true}){
   for(int i=0;i<q;i++)check[i]=(right?std::upper_bound(a.begin(),a.end(),keys[i]):std::lower_bound(a.begin(),a.end(),keys[i]))-a.begin();
   search<0>(a,keys,o,right);if(o!=check)return 10;
   search<1>(a,keys,o,right);if(o!=check)return 11;
   search<2>(a,keys,o,right);if(o!=check)return 12;
   cases++;
  }
 }
 std::cout<<"correctness_pass="<<cases<<"\\n";
 for(int n:{128,4096,65536,1048576})for(int q:{1,16,1024,65536})for(int pattern=0;pattern<3;pattern++)for(bool right:{false,true}){
  std::vector<i64>a(n),keys(q),out(q);
  for(int i=0;i<n;i++)a[i]=i;
  for(auto&k:keys)k=(i64)(gen()%(n+n/5))-n/10;
  if(pattern==1)std::sort(keys.begin(),keys.end());
  if(pattern==2)std::fill(keys.begin(),keys.end(),keys[0]);
  double timings[3]{};
  for(int repeat=0;repeat<5;repeat++)for(int mode: {0,1,2}){
   const int loops=std::max(1,std::min(1000,1000000/q));
   auto start=std::chrono::steady_clock::now();
   for(int iter=0;iter<loops;iter++){
    if(mode==0)search<0>(a,keys,out,right);
    if(mode==1)search<1>(a,keys,out,right);
    if(mode==2)search<2>(a,keys,out,right);
   }
   auto end=std::chrono::steady_clock::now();
   double elapsed=std::chrono::duration<double,std::nano>(end-start).count()/loops;
   timings[mode]+=elapsed/5.;
   sink=sink+out[0];
  }
  std::cout<<"case n="<<n<<" q="<<q<<" pattern="<<pattern<<" right="<<right<<" base_ns="<<timings[0]<<" a0_ns="<<timings[1]<<" bit_ns="<<timings[2]<<" a0_speedup="<<timings[0]/timings[1]<<" bit_speedup="<<timings[0]/timings[2]<<"\\n";
 }
}
