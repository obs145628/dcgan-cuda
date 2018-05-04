#pragma once

#include "op.hh"

namespace ops
{

    class SoftmaxCrossEntropy: public Op
    {

    public:

	SoftmaxCrossEntropy(Op* y, Op* logits);

	virtual void compile() override;	
    };

}
