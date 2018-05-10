#include "activ.hh"
#include "../ops/ops-builder.hh"
#include "../ops/vect-sigmoid.hh"

ops::Op* sigmoid(ops::Op* x)
{
    auto& builder = ops::OpsBuilder::instance();
    return builder.vect_sigmoid(x);
}
