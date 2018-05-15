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

    
    Tensor4 y = conv_c1f1i1snp0(x, k, 5, 3);
    Tensor4 dk = conv_dk_c1f1i1snp0(x, dy, 5, 3);
    Tensor4 dx = conv_dx_c1f1i1snp0(k, dy, 5, 3);

    std::vector<Tensor4> output {y, dk, dx};
    std::cout << "== output:\n";
    for (auto& t : output)
        t.dump_shape();
    std::cout << "==\n\n";
    Tensor4::save_tensors("./out.tbin", output);
    

    
}
