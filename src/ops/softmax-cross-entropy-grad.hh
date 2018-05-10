#pragma once

#include "op.hh"

namespace ops
{

    class SoftmaxCrossEntropyGrad: public Op
    {

    public:

        SoftmaxCrossEntropyGrad(Op* y, Op* logits);

        virtual void compile() override;
    };

}
