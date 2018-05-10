#pragma once

#include "op.hh"

namespace ops
{

    class SigmoidCrossEntropy: public Op
    {

    public:

        /**
         * Sigmoid cross entropy loss function
         * reduce_mean(cross_entropy(y, sigmoid(logits)))
         */
        SigmoidCrossEntropy(Op* y, Op* logits);

        virtual void compile() override;

        virtual Op* child_grad(std::size_t index, Op* dout) override;
    };

}
