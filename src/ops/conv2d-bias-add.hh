#pragma once

#include "op.hh"

namespace ops
{

    class Conv2DBiasAdd : public Op
    {
    public:
        Conv2DBiasAdd(Op* z, Op* bias);

        virtual void compile() override;

        virtual Op* child_grad(std::size_t index, Op* dout) override;
    };
}
