#pragma once

#include "fwd.hh"
#include "initializer.hh"
#include "../ops/op.hh"

struct DenseLayerData
{
    ops::Op* w;
    ops::Op* b;
    ops::Op* z;
};

struct Conv2DLayerData
{
  ops::Op* w;
  ops::Op* b;
  ops::Op* z;
};

struct Conv2DTransposeLayerData
{
  ops::Op* w;
  ops::Op* b;
  ops::Op* z;
};


ops::Op* dense_layer(ops::Op* input,
                     std::size_t in_size,
                     std::size_t out_size,
                     activ_f activ = nullptr,
                     Initializer* w_init = nullptr,
                     Initializer* b_init = nullptr,
                     DenseLayerData* tmp_data = nullptr);

ops::Op* conv2d_layer(ops::Op* input,
                      std::size_t nb_filter,
                      std::size_t* kernel_size,
                      int* strides,
                      std::size_t* in_size,
                      activ_f activ,
                      Initializer* w_init,
                      Initializer* b_init,
                      Conv2DLayerData* tmp_data);

ops::Op* conv2d_transpose_layer(ops::Op* input,
                                std::size_t nb_filter,
                                std::size_t* kernel_size,
                                int* strides,
                                std::size_t* in_size,
                                activ_f activ,
                                Initializer* w_init,
                                Initializer* b_init,
                                Conv2DTransposeLayerData* tmp_data);
