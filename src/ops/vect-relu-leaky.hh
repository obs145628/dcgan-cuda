#pragma once

#include "op.hh"

namespace ops
{

    class VectReluLeaky : public Op
    {
    public:
        VectReluLeaky(Op* arg, const dbl_t alpha);

        virtual void compile() override;

    private:
        const dbl_t alpha;
    };

}
