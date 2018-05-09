#pragma once

#include "op.hh"

namespace ops
{

    /**
     * Dense layer operation
     * output = dot(x, w) + b
     * x - input matrix (m * n)
     * w - weights matrix (n * p)
     * b - biais vector (p)
     * output matrix (m * p)
     */
    class MatMulAdd : public Op
    {

    public:
        MatMulAdd(Op* x, Op* w, Op* b);

        virtual void compile() override;

        virtual Op* child_grad(std::size_t index, Op* dout) override;
        
    };
    
}
