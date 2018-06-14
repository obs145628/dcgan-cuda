#pragma once

#include "../runtime/fwd.hh"

namespace gpu
{
    void kernel_conv2d(rt::Node* node);
    void kernel_conv2d_input_grad(rt::Node* node);
    void kernel_conv2d_kernel_grad(rt::Node* node);
    void kernel_conv2d_transpose(rt::Node* node);
    void kernel_conv2d_transpose_input_grad(rt::Node* node);
}
