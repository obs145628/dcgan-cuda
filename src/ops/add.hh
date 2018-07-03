#pragma once

#include "op.hh"

namespace ops
{

    class Add : public Op
    {
    public:
        Add(Op* left, Op* right);

        virtual void compile() override;

        virtual Op* child_grad(std::size_t index, Op* dout) override;
    };
    
}
