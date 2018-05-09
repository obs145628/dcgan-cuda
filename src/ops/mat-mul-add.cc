#include "mat-mul-add.hh"
#include <cassert>
#include <stdexcept>
#include "graph.hh"
#include "mse-grad.hh"
#include "ops-builder.hh"
#include "../runtime/node.hh"
#include "../memory/alloc.hh"

namespace ops
{

    MatMulAdd::MatMulAdd(Op* x, Op* w, Op* b)
        : Op(Shape({x->shape_get()[0], w->shape_get()[1]}), {x, w, b})
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

    /*
    Op* MSE::child_grad(std::size_t index, Op* dout)
    {
        assert(index < 2);
        if (index == 0)
            throw std::runtime_error {"Can't compute gradient of MSE for y"};

        if (dout != nullptr)
            throw std::runtime_error {"MSE must be the final node of the gradient"};


        auto& builder = OpsBuilder::instance();
        return builder.mse_grad(preds()[0] , preds()[1]);
    }
    */
    
}
