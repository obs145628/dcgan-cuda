#pragma once

#include "op.hh"

namespace ops
{

    class Conv2DTranspose : public Op
    {
    public:
        Conv2DTranspose(Op* input, Op* kernel, const int* out_size, const int* strides);

        virtual void compile() override;

        virtual Op* child_grad(std::size_t index, Op* dout) override;
    private:
        const int* m_out_size;
        const int* m_strides;
        const ops::Shape m_input_shape;
        const ops::Shape m_kernel_shape;
    };
}
