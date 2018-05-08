#pragma once

#include "op.hh"

namespace ops
{

    class VectReluLeaky : public Op
    {
    public:
        VectReluLeaky(Op* arg);

        virtual void compile() override;

    };

}
