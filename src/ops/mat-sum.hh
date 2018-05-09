#pragma once

#include "op.hh"

namespace ops
{

    class MatSum : public Op
    {

    public:
        MatSum(Op* arg, std::size_t axis);

        virtual void compile() override;

    private:
        std::size_t axis_;
    };
    
}
