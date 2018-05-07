#pragma once

#include "op.hh"

namespace ops
{

    class Input : public Op
    {
    public:
        Input(const Shape& shape);

        std::size_t input_id() const;

        virtual void compile() override;

    private:
        std::size_t id_;
        dbl_t* data_;
    };
    
}
