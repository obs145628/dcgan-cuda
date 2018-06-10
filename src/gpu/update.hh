#pragma once

#include "../runtime/fwd.hh"

namespace gpu
{
    void kernel_update(rt::Node* node);
    void kernel_moment_update(rt::Node* node);
    void kernel_moment_update2(rt::Node* node);
    void kernel_adam_update(rt::Node* node);
}
