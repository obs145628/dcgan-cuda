#include "mat-mul-add.hh"
#include <cassert>
#include <stdexcept>
#include "graph.hh"
#include "mat-mat-mul.hh"
#include "mat-sum.hh"
#include "ops-builder.hh"
#include "../runtime/node.hh"
#include "../memory/alloc.hh"

namespace ops
{

    MatMulAdd::MatMulAdd(Op* x, Op* w, Op* b)
        : Op("mat_mul_add",
             Shape({x->shape_get()[0], w->shape_get()[1]}), {x, w, b})
    {}

    void MatMulAdd::compile()
    {
        auto& g = Graph::instance();

        auto& cx = g.compiled(preds()[0]);
        auto& cw = g.compiled(preds()[1]);
        auto& cb = g.compiled(preds()[2]);

        std::size_t rows = cx.out_shape[0];
        std::size_t cols = cx.out_shape[1];
        std::size_t out_cols = cw.out_shape[1]; 
        Shape out_shape ({int(rows), int(out_cols)});
        dbl_t* out_data = tensor_alloc(out_shape.total());

        auto out_node = rt::Node::op_mat_mul_add(cx.out_data, cw.out_data, cb.out_data, out_data,
                                                 rows, cols, out_cols,
                                                 {cx.out_node, cw.out_node, cb.out_node});

        g.add_compiled(this, {out_node}, {out_data}, out_node, out_shape, out_data);
    }


    Op* MatMulAdd::child_grad(std::size_t index, Op* dout)
    {
        assert(index < 3);

        if (dout == nullptr)
            throw std::runtime_error {"MatMullAdd can't be the final node of the gradient"};


        auto& builder = OpsBuilder::instance();
        auto x = preds()[0];
        auto w = preds()[1];
        auto b = preds()[2];
        (void) b;

        // dC/dx
        if (index == 0)
            return builder.mat_mat_mul(dout, w, false, true);

        // dC/dw
        else if (index == 1)
            return builder.mat_mat_mul(x, dout, true, false);

        // dC/db
        else
            return builder.mat_sum(dout, 0);
    }
    
}
