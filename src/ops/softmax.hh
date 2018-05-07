#pragma once

#include "op.hh"

namespace ops
{
    class Softmax : public Op
    {
    public:
        Softmax(Op* arg);

        virtual void compile() override;

    };
}
