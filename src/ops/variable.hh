#pragma once

#include "op.hh"
#include "../memory/types.hh"

namespace ops
{

    class Variable : public Op
    {
    public:
        Variable(const Shape& shape,
                 bool trainable);
        virtual ~Variable();

        std::size_t var_id() const;

        void compile() override;

        void write(const dbl_t* ptr);
        void read(dbl_t* ptr) const;
        bool is_trainable() const;

        dbl_t* data_begin();
        dbl_t* data_end();

    private:
        dbl_t* data_;
        Shape real_shape_;
        bool trainable_;
        std::size_t id_;

    };
    
    
}
