#pragma once

#include "op.hh"

namespace ops
{

    class SigmoidGrad : public Op
    {

    public:
        SigmoidGrad(Op* sig_out, Op* dout);

        virtual void compile() override;

    };
    
}
