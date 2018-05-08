#pragma once

#include "op.hh"

namespace ops
{

    class VectTanh : public Op
    {
    public:
        VectTanh(Op* arg);

        virtual void compile() override;

    };

}
