#pragma once

#include "op.hh"

namespace ops
{

    class Conv2DBiasAdd : public Op
    {
    public:
        Conv2DBiasAdd(Op* z, Op* bias);

        virtual void compile() override;
    };
}