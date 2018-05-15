#include <iostream>

#include "tensor4.hh"
#include "conv.hh"
#include "conv_simplified.hh"

int main()
{
    auto input = Tensor4::load_tensors("./conv_in.npz");
    std::cout << "== input:\n"; 
    for (auto& t : input)
        t.dump_shape();
    std::cout << "==\n\n";

    const Tensor4& x = input[0];
    const Tensor4& k = input[1];
    const Tensor4& dy = input[2];
    (void) k;
    (void) dy;

    std::size_t sh = 2;
    std::size_t sw = 2;
    std::size_t p1 = 1;
    std::size_t p2 = 2;
    std::size_t p3 = 1;
    std::size_t p4 = 2;
    
    Tensor4 y = conv_cnfni1snpn(x, k, sh, sw, p1, p2, p3, p4);
    Tensor4 dk = conv_dk_cnfni1snpn(x, dy, sh, sw, p1, p2, p3, p4);
    Tensor4 dx = conv_dx_cnfni1snpn(k, dy, sh, sw, p1, p2, p3, p4);

    std::vector<Tensor4> output {y, dk, dx};
    std::cout << "== output:\n";
    for (auto& t : output)
        t.dump_shape();
    std::cout << "==\n\n";
    Tensor4::save_tensors("./out.tbin", output);
    

    
}
