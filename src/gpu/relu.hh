#include "../runtime/fwd.hh"

namespace gpu
{
    void kernel_relu(rt::Node* node);
    void kernel_relu_grad(rt::Node* node);
    void kernel_relu_leaky(rt::Node* node);
    void kernel_leaky_relu_grad(rt::Node* node);
}
