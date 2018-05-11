#pragma once

#include "op.hh"

namespace ops
{

    class Update : public Op
    {
    public:

        /**
         * Update the value of the variable var
         * var = var + coeff * dt
         * var and dt have the same shape
         * coeff must be a scalar
         */
        Update(Variable* var, Op* dt, Op* coeff);

        virtual void compile() override;

    private:
        Variable* var_;
    };
    
}
