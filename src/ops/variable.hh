#pragma once

#include "op.hh"
#include "../memory/types.hh"

namespace ops
{

    class Variable : public Op
    {
    public:
        Variable(const Shape& shape);
        virtual ~Variable();

        std::size_t var_id() const;

        void compile() override;

        void write(const dbl_t* ptr);
        void read(dbl_t* ptr) const;

    private:
        dbl_t* data_;
        Shape real_shape_;
        std::size_t id_;

    };
    
    
}
