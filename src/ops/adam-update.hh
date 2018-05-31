#pragma once

#include "op.hh"

namespace ops
{

    class AdamUpdate : public Op
    {
    public:

        /**
         * Update the value of the variable var
         * For the Adam Optimiser
         *
         * t <- t + 1
         * lrt <- lr * sqrt(1 - beta2^t) / (1 - beta1^t)
         * var = var - lrt * m / (sqrt(v) + eps)
         *
         * var and m and v have the same shape
         * lr, beta1, beta2 and eps are constants
         */
        AdamUpdate(Variable* var, Op* m, Op* v,
                   dbl_t learning_rate,
                   dbl_t beta1, dbl_t beta2, dbl_t eps);
        ~AdamUpdate();

        virtual void compile() override;

    private:
        Variable* var_;
        dbl_t t_;
        dbl_t lr_;
        dbl_t beta1_;
        dbl_t beta2_;
        dbl_t eps_;
    };
    
}
