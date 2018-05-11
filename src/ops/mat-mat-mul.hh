#pragma once

#include "op.hh"

namespace ops
{
    /**
     * Perform a matrix-matrix multiplication
     * Handle cases of transpose
     */
    class MatMatMul : public Op
    {

    public:
        MatMatMul(Op* left, Op* right,
                  bool left_tr = false, bool right_tr = false);

        virtual void compile() override;

    private:
        bool left_tr_;
        bool right_tr_;
    };
    
}
