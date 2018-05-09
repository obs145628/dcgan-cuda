#include "mat-sum.hh"
#include <cassert>
#include <stdexcept>
#include "graph.hh"
#include "ops-builder.hh"
#include "sigmoid-grad.hh"
#include "../runtime/node.hh"
#include "../memory/alloc.hh"

namespace ops
{

    MatSum::MatSum(Op* arg, std::size_t axis)
        : Op(Shape({arg->shape_get()[!axis]}), {arg})
        , axis_(axis)
    {}

    void MatSum::compile()
    {
        auto& g = Graph::instance();
        auto& carg = g.compiled(preds()[0]);

        std::size_t rows = carg.out_shape[0];
        std::size_t cols = carg.out_shape[1];
        Shape out_shape = Shape({carg.out_shape[!axis_]});
        dbl_t* out_data = tensor_alloc(out_shape.total());

        
        auto out_node = axis_ == 0 ?
            rt::Node::op_mat_sum_rows(carg.out_data, out_data,
                                      rows, cols,
                                      {carg.out_node})
            :
            nullptr;
        assert(out_node);

        g.add_compiled(this, {out_node}, {out_data}, out_node, out_shape, out_data);
    }
}
