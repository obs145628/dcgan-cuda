#pragma once

#include "op.hh"

namespace ops
{

    class VectSigmoid : public Op
    {
    public:
        VectSigmoid(Op* arg);

        virtual void compile() override;

        virtual Op* child_grad(std::size_t index, Op* dout) override;
    };
    
}
