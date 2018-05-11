#pragma once

#include "op.hh"

namespace ops
{

    class Reshape : public Op
    {
    public:
        Reshape(Op* arg, const Shape& shape);

        virtual void compile() override;
    };

}
