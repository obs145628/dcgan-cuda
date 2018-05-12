#include "layers.hh"
#include "normal-initializer.hh"
#include "zero-initializer.hh"
#include "../ops/ops-builder.hh"
#include "../ops/mat-mat-mul.hh"
#include "../ops/mat-rvect-add.hh"
#include "../ops/mat-mul-add.hh"
#include "../ops/variable.hh"
#include "../ops/vect-sigmoid.hh"
#include "../ops/conv2d.hh"
#include "../ops/conv2d-bias-add.hh"


ops::Op* dense_layer(ops::Op* input,
                     std::size_t in_size,
                     std::size_t out_size,
                     activ_f activ,
                     Initializer* w_init,
                     Initializer* b_init,
                     DenseLayerData* tmp_data)
{
    auto& builder = ops::OpsBuilder::instance();
    
    auto w = builder.variable(ops::Shape({int(in_size), int(out_size)}), true);
    w->extend_name("dense_w");
    auto b = builder.variable(ops::Shape({int(out_size)}), true);
    b->extend_name("dense_b");

    NormalInitializer w_base_init;
    ZeroInitializer b_base_init;
    if (!w_init)
        w_init = &w_base_init;
    if (!b_init)
        b_init = &b_base_init;

    w_init->fill(w->data_begin(), w->data_end());
    b_init->fill(b->data_begin(), b->data_end());
    
    ops::Op* out = builder.mat_mul_add(input, w, b);
    ops::Op* z = out;
    if (activ)
    {
        out = activ(z);
        out->extend_name("dense_activ");
    }

    if (tmp_data)
    {
        tmp_data->w = w;
        tmp_data->b = b;
        tmp_data->z = z;
    }
    
    return out;
}

ops::Op* conv2d_layer(ops::Op* input,
                      std::size_t nb_filter,
                      std::size_t* kernel_size,
                      int* strides,
                      std::size_t* in_size,
                      activ_f activ,
                      Initializer* w_init,
                      Initializer* b_init,
                      Conv2DLayerData* tmp_data)
{
    auto& builder = ops::OpsBuilder::instance();
   
    auto w = builder.variable(ops::Shape({int(kernel_size[0]), int(kernel_size[1]),
                    int(in_size[3]), int(nb_filter)}), true);
    w->extend_name("conv2d_w");
    auto b = builder.variable(ops::Shape({int(nb_filter)}), true);
    b->extend_name("conv2d_b");

    NormalInitializer base_init;
    if (!w_init)
        w_init = &base_init;
    if (!b_init)
        b_init = &base_init;

    w_init->fill(w->data_begin(), w->data_end());
    b_init->fill(b->data_begin(), b->data_end());
    
    ops::Op* out_conv = builder.conv2d(input, w, strides);
    ops::Op* out = builder.conv2d_bias_add(out_conv, b);
    ops::Op* z = out;
    if (activ)
    {
        out = activ(z);
        out->extend_name("conv2d_activ");
    }
    
    if (tmp_data)
    {
        tmp_data->w = w;
        tmp_data->b = b;
        tmp_data->z = z;
    }
    
    return out;
}
