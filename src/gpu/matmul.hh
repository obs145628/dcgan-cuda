#pragma once

#include "../runtime/fwd.hh"

namespace gpu
{
    void kernel_mat_mat_mul(rt::Node* node);
    void kernel_mat_rvect_add(rt::Node* node);
    void kernel_mat_mul_add(rt::Node* node);
    void kernel_tmat_mat_mul(rt::Node* node);
    void kernel_mat_tmat_mul(rt::Node* node);
}
