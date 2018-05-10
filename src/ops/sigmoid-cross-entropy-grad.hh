#pragma once

#include "op.hh"

namespace ops
{

    class SigmoidCrossEntropyGrad: public Op
    {

    public:

        SigmoidCrossEntropyGrad(Op* y, Op* logits);

        virtual void compile() override;
    };

}
