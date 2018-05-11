#pragma once

#include "op.hh"

namespace ops
{

    class MSEGrad : public Op
    {

    public:
        MSEGrad(Op* y, Op* y_hat);

        virtual void compile() override;
    };
}
