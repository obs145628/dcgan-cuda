#pragma once

#include "fwd.hh"
#include "../ops/op.hh"

struct DenseLayerData
{
    ops::Op* w;
    ops::Op* b;
    ops::Op* z;
};


ops::Op* dense_layer(ops::Op* input,
                     std::size_t in_size,
                     std::size_t out_size,
                     activ_f activ = nullptr,
                     dbl_t* w_init = nullptr,
                     dbl_t* b_init = nullptr,
                     DenseLayerData* tmp_data = nullptr);


