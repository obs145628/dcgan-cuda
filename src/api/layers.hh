#pragma once

#include "../ops/op.hh"


ops::Op* dense_layer(ops::Op* input,
		     std::size_t in_size,
		     std::size_t out_size,
		     dbl_t* w_init = nullptr,
		     dbl_t* b_init = nullptr);


