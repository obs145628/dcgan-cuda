#pragma once

#include "../runtime/fwd.hh"

namespace gpu
{
    void kernel_softmax(rt::Node* node);
    void kernel_log_softmax(rt::Node* node);
}
