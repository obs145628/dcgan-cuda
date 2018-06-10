#include <iostream>
#include <cmath>

#include <tocha/tensor.hh>

#include "../src/memory/mode.hh"

#include "../src/utils/xorshift.hh"

#include "../src/memory/types.hh"


int main()
{

    xorshift::seed(234);
    
    constexpr std::size_t len = 145*18*12*34; 
    dbl_t data[len];
    xorshift::fill(data, data + len);

    tocha::Tensors out;
    out.add(tocha::Tensor::f32(145, 18, 12, 34));
    dbl_t* y_out = reinterpret_cast<dbl_t*>(out.arr()[0].data);
    for (std::size_t i = 0; i < len; ++i)
        y_out[i] = data[i];
    
    out.save("./out.tbin");
    
}
