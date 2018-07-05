#pragma once

#include "shape.hh"
#include <string>

#include "compiled-op.hh"

namespace ops
{

    class Op
    {

    public:
        /**
         * Graph operation
         * name - name of the node, for debugging purposes only
         * shape - shape of the result of the operation
         * preds - inputs of the operation
         * succs - every node whose input contains this node
         */
        Op(const std::string& name,
           const Shape& shape,
           const std::vector<Op*> preds = {},
           const std::vector<Op*> succs = {});
        virtual ~Op() = default;

        const std::string& name_get();
        void extend_name(const std::string& str);
        const Shape& shape_get() const;
        std::vector<Op*> preds();
        std::vector<Op*> succs();

        /**
         * Allocate necessary memory
         * Create equivalent rt::Node objects fort the operation
         */
        virtual void compile() = 0;

        /**
         * Return the index of pred in the lists of inputs
         * or -1 if not found
         */
        std::size_t pred_index(const Op* pred);

        /**
         * Find which of the succesors of the node is a direct or undirect predecessor of node
         */
        Op* pred_of(const Op* node);

        /**
         * Find all of the succesors of the node that are a direct or undirect predecessor of node
         */
        std::vector<Op*> preds_of(const Op* node);
        

        /**
         * Compute the gradient nabla(C) / nabla(i-th child)
         * C: any node that outputs a number
         * dout: nabla(C) / nabla(this)
         * if C = this, then dout is nullptr
         */
        virtual Op* child_grad(std::size_t index, Op* dout);

    private:
        std::string name_;
        Shape shape_;
        std::vector<Op*> preds_;
        std::vector<Op*> succs_;
    };
}
