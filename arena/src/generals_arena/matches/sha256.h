// Small streaming SHA-256 for portable replay digests (FIPS 180-4).
#pragma once
#include <array>
#include <cstdint>
#include <string>
struct SHA256 {
    std::uint32_t h[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    std::uint8_t block[64]{};std::uint64_t size=0;int used=0;
    static std::uint32_t r(std::uint32_t v,int n){return (v>>n)|(v<<(32-n));}
    void compress(){
        static constexpr std::uint32_t k[64]={
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        std::uint32_t w[64];
        for(int i=0;i<16;++i)w[i]=(std::uint32_t(block[4*i])<<24)|(std::uint32_t(block[4*i+1])<<16)|(std::uint32_t(block[4*i+2])<<8)|block[4*i+3];
        for(int i=16;i<64;++i)w[i]=w[i-16]+(r(w[i-15],7)^r(w[i-15],18)^(w[i-15]>>3))+w[i-7]+(r(w[i-2],17)^r(w[i-2],19)^(w[i-2]>>10));
        auto a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],v=h[7];
        for(int i=0;i<64;++i){auto t=v+(r(e,6)^r(e,11)^r(e,25))+((e&f)^(~e&g))+k[i]+w[i];
            auto u=(r(a,2)^r(a,13)^r(a,22))+((a&b)^(a&c)^(b&c));v=g;g=f;f=e;e=d+t;d=c;c=b;b=a;a=t+u;}
        std::uint32_t values[8]={a,b,c,d,e,f,g,v};for(int i=0;i<8;++i)h[i]+=values[i];
    }
    void add(const void *data,std::size_t length){auto p=static_cast<const std::uint8_t *>(data);size+=length;
        for(std::size_t i=0;i<length;++i){block[used++]=p[i];if(used==64){compress();used=0;}}}
    void add(const std::string &s){add(s.data(),s.size());}
    template<class T> void integers(const T *p,std::size_t n){for(std::size_t i=0;i<n;++i){
        auto v=std::uint64_t(std::int64_t(p[i]));std::uint8_t bytes[8];for(int j=0;j<8;++j)bytes[j]=v>>(j*8);add(bytes,8);}}
    std::string finish(){auto bits=size*8;std::uint8_t one=128,zero=0;add(&one,1);
        while(used!=56)add(&zero,1);std::uint8_t tail[8];for(int i=0;i<8;++i)tail[7-i]=bits>>(i*8);add(tail,8);
        std::string out;const char *hex="0123456789abcdef";for(auto v:h)for(int j=7;j>=0;--j)out+=hex[(v>>(j*4))&15];return out;}
};
