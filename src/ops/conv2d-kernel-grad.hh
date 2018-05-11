#pragma once

#include "op.hh"

namespace ops
{

    class Conv2DKernelGrad : public Op
    {
    public:
        Conv2DKernelGrad(Op* y, Op* input, const int* strides, const int* kernel_size);

        virtual void compile() override;
    private:
        const int* m_strides;
        int m_kernel_size[4];
    };
}
