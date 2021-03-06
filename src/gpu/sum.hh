#pragma once

#include "../runtime/fwd.hh"

namespace gpu
{
    void kernel_mse(rt::Node* node);
    void kernel_mse_grad(rt::Node* node);
    void kernel_mat_sum_rows(rt::Node* node);
    void kernel_mat_sum_cols(rt::Node* node);
    void kernel_argmax_acc(rt::Node* node);
    void kernel_add(rt::Node* node);
}
