#pragma once

#include "op.hh"

namespace ops
{
    class LogSoftmax : public Op
    {
    public:
        LogSoftmax(Op* arg);

        virtual void compile() override;

    };
}
