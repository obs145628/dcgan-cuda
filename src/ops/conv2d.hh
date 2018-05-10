#pragma once

#include "op.hh"

namespace ops
{

    class Conv2D : public Op
    {
    public:
        Conv2D(Op* input, Op* kernel, const int* strides);

        virtual void compile() override;
        
        virtual Op* child_grad(std::size_t index, Op* dout) override;
    private:
        const int* m_strides;
    };
}
