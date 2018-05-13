#pragma once

#include "op.hh"

namespace ops
{

    class LeakyReluGrad : public Op
    {

    public:

        /**
         * Gradient of the leaky-relu operation
         * z - input of the leaky-relu node
         * dout - gradient of leaky-relu(z)
         */
        LeakyReluGrad(Op* z, Op* dout, dbl_t alpha);

        virtual void compile() override;

    private:
        dbl_t alpha_;
    };
    
}
