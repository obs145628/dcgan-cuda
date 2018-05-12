#pragma once

#include "op.hh"

namespace ops
{

    class Conv2DInputGrad : public Op
    {
    public:
        Conv2DInputGrad(Op* y, Op* kernel, const int* strides, const int* input_size);

        virtual void compile() override;
    private:
        const int* m_strides;
        int m_input_size[4];
    };
}
