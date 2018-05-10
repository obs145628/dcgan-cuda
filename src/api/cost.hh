#pragma once

#include "../ops/op.hh"

ops::Op* quadratic_cost(ops::Op* y, ops::Op* y_hat);
ops::Op* softmax_cross_entropy(ops::Op* y, ops::Op* logits);
