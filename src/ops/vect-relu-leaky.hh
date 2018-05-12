#pragma once

#include "op.hh"

namespace ops
{

    class VectReluLeaky : public Op
    {
    public:
        VectReluLeaky(Op* arg, const dbl_t alpha);

        virtual void compile() override;
        virtual Op* child_grad(std::size_t index, Op* dout) override;

    private:
        const dbl_t alpha;
    };

}
