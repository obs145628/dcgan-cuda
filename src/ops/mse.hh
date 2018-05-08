#pragma once

#include "op.hh"

namespace ops
{

    class MSE: public Op
    {

    public:

        MSE(Op* y, Op* y_hat);

        virtual void compile() override;

        virtual Op* child_grad(std::size_t index, Op* dout) override;
    };

}
