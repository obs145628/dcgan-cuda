#pragma once

#include "op.hh"

namespace ops
{

    class MomentUpdate : public Op
    {
    public:

        /**
         * Update the value of the variable var
         * var = coeff1 * var + coeff2 * dt
         * var = coeff1 * var + coeff2 * dt * dt
         * var and dt have the same shape
         * coeff1 and coeff2 are scalars
         */
        MomentUpdate(Variable* var, Op* dt,
                     dbl_t coeff1, dbl_t coeff2, bool sq_update);

        virtual void compile() override;

    private:
        Variable* var_;
        dbl_t coeff1_;
        dbl_t coeff2_;
        bool sq_update_;
    };
    
}
