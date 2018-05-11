#pragma once

#include "op.hh"

namespace ops
{

    class ReluGrad : public Op
    {

    public:

        /**
         * Gradient of the relu operation
         * z - input of the relu node
         * dout - gradient of relu(z)
         */
        ReluGrad(Op* z, Op* dout);

        virtual void compile() override;

    };
    
}
