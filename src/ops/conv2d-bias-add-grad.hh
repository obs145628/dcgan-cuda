#pragma once

#include "op.hh"

namespace ops
{

    class Conv2DBiasAddGrad : public Op
    {
    public:
        Conv2DBiasAddGrad(Op* dz);

        virtual void compile() override;
    };
}
