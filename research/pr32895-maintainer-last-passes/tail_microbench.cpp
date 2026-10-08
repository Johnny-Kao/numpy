// Standalone controlled microbenchmark: faithful integer batched-search loops.
// Research evidence only, not an end-to-end NumPy performance claim.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
#include <cstdlib>
using i64 = std::int64_t;
using idx = std::ptrdiff_t;
static volatile idx sink=0;
void baseline(const i64* a,const i64* key,idx* out,idx n,idx q) {
  if(n<=0) { std::fill(out,out+q,0); return; }
  idx len=n,half=len>>1; len-=half;
  const i64 pivot=a[half];
  for(idx i=0;i<q;++i) out[i]=(pivot<key[i])*half;
  while(len>1) {
    half=len>>1;len-=half;
    for(idx i=0;i<q;++i) {idx &b=out[i]; b+=(a[b+half]<key[i])*half;}
  }
  for(idx i=0;i<q;++i) {idx &b=out[i];b+=(a[b]<key[i]);}
}
void tail2(const i64* a,const i64* key,idx* out,idx n,idx q) {
  if(n<=0) { std::fill(out,out+q,0); return; }
  idx len=n,half=len>>1;len-=half;
  const i64 pivot=a[half];
  for(idx i=0;i<q;++i) out[i]=(pivot<key[i])*half;
  while(len>2) {
    half=len>>1;len-=half;
    for(idx i=0;i<q;++i) {idx &b=out[i];b+=(a[b+half]<key[i])*half;}
  }
  for(idx i=0;i<q;++i) {
    idx &b=out[i]; const i64 k=key[i]; idx remaining=len;
    while(remaining>1) {
      const idx step=remaining>>1;remaining-=step;
      b+=(a[b+step]<k)*step;
    }
    b+=(a[b]<k);
  }
}
int main(){
 std::mt19937_64 rng(32895);
 for(idx n:{idx(32),idx(256),idx(4096),idx(65536)}){
 for(idx q:{idx(64),idx(4096),idx(65536)}){
  std::vector<i64>a(n),keys(q),out0(q),out1(q);
  for(idx i=0;i<n;++i)a[i]=i*3;
  for(const char* pattern:{"random","ascending","same"}){
   for(idx i=0;i<q;++i)keys[i]=(i64)(rng()%(3*n+10));
   if(pattern[0]=='a')std::sort(keys.begin(),keys.end());
   if(pattern[0]=='s')std::fill(keys.begin(),keys.end(),a[n/2]);
   baseline(a.data(),keys.data(),out0.data(),n,q);
   tail2(a.data(),keys.data(),out1.data(),n,q);
   if(out0!=out1){std::fprintf(stderr,"MISMATCH n=%td q=%td pattern=%s\n",n,q,pattern);return 2;}
   constexpr int rounds=12;
   double ms[2]={0,0};
   for(int pass=0;pass<2;++pass) {
    const int which=pass;
    auto fn=which?tail2:baseline;
    for(int r=0;r<rounds;++r){
     auto t0=std::chrono::steady_clock::now();
     fn(a.data(),keys.data(),out1.data(),n,q);
     auto t1=std::chrono::steady_clock::now();
     ms[which]+=std::chrono::duration<double,std::micro>(t1-t0).count();
     sink=sink+out1[r%q];
    }
   }
   std::printf("N=%td Q=%td pattern=%s baseline_us=%.3f tail2_us=%.3f speedup=%.3f\n",n,q,pattern,ms[0]/rounds,ms[1]/rounds,ms[0]/ms[1]);
  }
 }}
}
