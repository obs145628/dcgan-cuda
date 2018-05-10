#pragma once

#include "op.hh"

namespace ops
{

    class VectRelu : public Op
    {
    public:
        VectRelu(Op* arg);

        virtual void compile() override;

        virtual Op* child_grad(std::size_t index, Op* dout) override;
    };

}
