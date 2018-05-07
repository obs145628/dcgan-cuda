#pragma once

#include "op.hh"

namespace ops
{

    class MatMatMul : public Op
    {

    public:
        MatMatMul(Op* left, Op* right);

        virtual void compile() override;
    };
    
}
