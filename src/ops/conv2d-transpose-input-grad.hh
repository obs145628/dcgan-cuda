#pragma once

#include "op.hh"

namespace ops
{

    class Conv2DTransposeInputGrad : public Op
    {
    public:
        Conv2DTransposeInputGrad(Op* y, Op* kernel, const int* strides, const int* input_size);

        virtual void compile() override;
    private:
        const int* m_strides;
        int m_input_size[4];
    };
}
