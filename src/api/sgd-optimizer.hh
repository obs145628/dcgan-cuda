#pragma once

#include "../memory/types.hh"
#include "../ops/fwd.hh"

class SGDOptimizer
{

public:
    SGDOptimizer(dbl_t learning_rate);

    ops::Op* minimize(ops::Op* loss);

private:
    dbl_t learning_rate_;
};
