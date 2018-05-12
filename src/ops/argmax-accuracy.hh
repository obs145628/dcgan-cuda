#pragma once

#include "op.hh"

namespace ops
{

    /**
     * Return the number of positive examples in y_hat
     * compute sum(equals(argmax(y), argmax(y_hat)))
     */
    class ArgmaxAccuracy: public Op
    {

    public:

        ArgmaxAccuracy(Op* y, Op* y_hat);

        virtual void compile() override;
    };

}
