#pragma once

#include "op.hh"

namespace ops
{

    class TanhGrad : public Op
    {

    public:
        TanhGrad(Op* sig_out, Op* dout);

        virtual void compile() override;

    };
    
}
