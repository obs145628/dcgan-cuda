#pragma once

#include "op.hh"

namespace ops
{

    class SoftmaxCrossEntropy: public Op
    {

    public:

        SoftmaxCrossEntropy(Op* y, Op* logits);

        virtual void compile() override;

        virtual Op* child_grad(std::size_t index, Op* dout) override;
    };

}
