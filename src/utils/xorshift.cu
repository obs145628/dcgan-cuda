#include "xorshift.hh"

namespace xorshift
{

    namespace
    {
        std::uint64_t s[4];
    }

    void seed(std::uint64_t x)
    {
        s[0] = 1000*x;
        s[1] = 2000*x;
        s[2] = 3000*x;
        s[3] = 4000*x;
    }
    
    std::uint64_t next_u64()
    {
        std::uint64_t t = s[0] ^ (s[0] << 11);
        s[0] = s[1];
        s[1] = s[2];
        s[2] = s[3];
        s[3] = s[3] ^ (s[3] >> 19) ^ t ^ (t >> 8);
        return s[3];
    }

    float next_f32()
    {
        float x = next_u64();
        float div = 0xFFFFFFFFFFFFFFFF;
        return x / div;
    }

    void fill(float* begin, float* end)
    {
        while (begin != end)
            *begin++ = next_f32();
    }
    

    /*
        std::uint64_t x = s[0];
        std::uint64_t y = s[1];
	s[0] = y;
	x ^= x << 23; // a
	s[1] = x ^ y ^ (x >> 17) ^ (y >> 26); // b, c
	return s[1] + y;
    */
}
