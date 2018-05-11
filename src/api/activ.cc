#include "activ.hh"
#include "../ops/ops-builder.hh"
#include "../ops/vect-relu-leaky.hh"
#include "../ops/vect-relu.hh"
#include "../ops/vect-sigmoid.hh"

ops::Op* leaky_relu(ops::Op* x)
{
    auto& builder = ops::OpsBuilder::instance();
    return builder.vect_relu_leaky(x);
}

ops::Op* relu(ops::Op* x)
{
    auto& builder = ops::OpsBuilder::instance();
    return builder.vect_relu(x);
}

ops::Op* sigmoid(ops::Op* x)
{
    auto& builder = ops::OpsBuilder::instance();
    return builder.vect_sigmoid(x);
}
