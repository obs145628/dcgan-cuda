#include "cost.hh"
#include "../ops/ops-builder.hh"
#include "../ops/softmax-cross-entropy.hh"
#include "../ops/mse.hh"

ops::Op* quadratic_cost(ops::Op* y, ops::Op* y_hat)
{
    auto& builder = ops::OpsBuilder::instance();

    return builder.mse(y, y_hat);
}

ops::Op* softmax_cross_entropy(ops::Op* y, ops::Op* logits)
{
    auto& builder = ops::OpsBuilder::instance();

    return builder.softmax_cross_entropy(y, logits);
}
