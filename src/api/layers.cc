#include "layers.hh"
#include "../ops/ops-builder.hh"
#include "../ops/mat-mat-mul.hh"
#include "../ops/mat-rvect-add.hh"
#include "../ops/mat-mul-add.hh"
#include "../ops/variable.hh"
#include "../ops/vect-sigmoid.hh"

ops::Op* dense_layer(ops::Op* input,
                     std::size_t in_size,
                     std::size_t out_size,
                     dbl_t* w_init,
                     dbl_t* b_init)
{
    auto& builder = ops::OpsBuilder::instance();
    
    auto w = builder.variable(ops::Shape({int(in_size), int(out_size)}));
    auto b = builder.variable(ops::Shape({int(out_size)}));

    if (w_init)
        w->write(w_init);
    if (b_init)
        b->write(b_init);
    
    ops::Op* z = builder.mat_mul_add(input, w, b);
    auto out = builder.vect_sigmoid(z);
    return out;
}
