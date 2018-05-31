#include "../runtime/fwd.hh"

namespace gpu
{
    void kernel_sigmoid(rt::Node* node);
    void kernel_sigmoid_grad(rt::Node* node);
    void kernel_tanh(rt::Node* node);
    void kernel_sigmoid_cross_entropy(rt::Node* node);
}
