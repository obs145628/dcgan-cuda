#include "layers.hh"
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

ops::Op* conv2d_layer(ops::Op* input,
                      std::size_t nb_filter,
                      std::size_t* kernel_size,
                      int* strides,
                      std::size_t* in_size,
                      activ_f activ,
                      dbl_t* w_init,
                      dbl_t* b_init,
                      Conv2DLayerData* tmp_data)
{
    auto& builder = ops::OpsBuilder::instance();
   
    auto w = builder.variable(ops::Shape({int(kernel_size[0]), int(kernel_size[1]),
                                         int(in_size[3]), int(nb_filter)}));
    auto b = builder.variable(ops::Shape({int(nb_filter)}));
   
    if (w_init)
      w->write(w_init);
    if (b_init)
      b->write(b_init);
    
    ops::Op* out_conv = builder.conv2d(input, w, strides);
    ops::Op* out = builder.conv2d_bias_add(out_conv, b);
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
