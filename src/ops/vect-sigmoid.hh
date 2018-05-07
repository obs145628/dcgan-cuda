#pragma once

#include "op.hh"

namespace ops
{

    class VectSigmoid : public Op
    {
    public:
        VectSigmoid(Op* arg);

        virtual void compile() override;

    };
    
}
