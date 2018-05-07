#pragma once

#include "op.hh"

namespace ops
{

    class MatRvectAdd : public Op
    {
    public:
        MatRvectAdd(Op* left, Op* right);

        void compile() override;
    };
    
}
