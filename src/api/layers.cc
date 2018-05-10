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
                     activ_f activ,
                     dbl_t* w_init,
                     dbl_t* b_init,
                     DenseLayerData* tmp_data)
{
    auto& builder = ops::OpsBuilder::instance();
    
    auto w = builder.variable(ops::Shape({int(in_size), int(out_size)}));
    auto b = builder.variable(ops::Shape({int(out_size)}));

    if (w_init)
        w->write(w_init);
    if (b_init)
        b->write(b_init);
    
    ops::Op* out = builder.mat_mul_add(input, w, b);
    ops::Op* z = out;
    if (activ)
        out = activ(z);

    if (tmp_data)
    {
        tmp_data->w = w;
        tmp_data->b = b;
        tmp_data->z = z;
    }
    
    return out;
}
