#pragma once

#include "../memory/types.hh"
#include "../ops/fwd.hh"

class AdamOptimizer
{

public:
    AdamOptimizer(dbl_t learning_rate=0.001,
                  dbl_t beta1=0.9,
                  dbl_t beta2=0.999,
                  dbl_t epsilon=1e-08);

    ops::Op* minimize(ops::Op* loss);

private:
    dbl_t lr_;
    dbl_t beta1_;
    dbl_t beta2_;
    dbl_t eps_;
};
