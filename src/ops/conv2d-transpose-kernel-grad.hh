#pragma once

#include "op.hh"

namespace ops
{

    class Conv2DTransposeKernelGrad : public Op
    {
    public:
        Conv2DTransposeKernelGrad(Op* y, Op* input, const int* strides, const int* kernel_size);

        virtual void compile() override;
    private:
        const int* m_strides;
        int m_kernel_size[4];
    };
}
