#include <iostream>

#include "tensor4.hh"
#include "conv.hh"

int main()
{

    auto input = Tensor4::load_tensors("./conv_in.npz");
    for (auto& t : input)
        t.dump_shape();

    Tensor4 res = conv_no_pad(input[0], input[1], 2, 3);
    res.dump_shape();

    Tensor4 res2 = conv_pad(input[0], input[1], 2, 3, 3, 5);

    res2.dump_shape();

    //10, 10, 5, 16

    Tensor4::save_tensors("./conv_out.tbin", {res2});
}
