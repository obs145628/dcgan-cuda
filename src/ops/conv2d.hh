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
        const ops::Shape m_input_shape;
        const ops::Shape m_kernel_shape;
        int m_pad_top = 0;
        int m_pad_left = 0;
        int m_padded_size[2];
    };
}
