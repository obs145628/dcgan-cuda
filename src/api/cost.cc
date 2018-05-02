#include "cost.hh"
#include "../ops/ops-builder.hh"
#include "../ops/mse.hh"

ops::Op* quadratic_cost(ops::Op* y, ops::Op* y_hat)
{
    auto& builder = ops::OpsBuilder::instance();

    return builder.mse(y, y_hat);
}
