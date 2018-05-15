#pragma once

#include "op.hh"

namespace ops
{

    class Reshape : public Op
    {
    public:
        Reshape(Op* arg, const Shape& shape);

        virtual void compile() override;

        virtual Op* child_grad(std::size_t index, Op* dout) override;
    private:
        const Shape& m_initial_size;
    };

}
