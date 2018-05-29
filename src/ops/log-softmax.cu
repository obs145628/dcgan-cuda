#include "log-softmax.hh"
#include "graph.hh"
#include "../runtime/node.hh"
#include "../memory/alloc.hh"

namespace ops
{

    LogSoftmax::LogSoftmax(Op* arg)
        : Op("log_softmax", arg->shape_get(), {arg})
    {}

    void LogSoftmax::compile()
    {
        auto& g = Graph::instance();
        auto& carg = g.compiled(preds()[0]);

        std::size_t rows = carg.out_shape[0];
        std::size_t cols = carg.out_shape[1];
        Shape out_shape = carg.out_shape;
        dbl_t* out_data = tensor_alloc(rows * cols);

        auto out_node = rt::Node::op_log_softmax(carg.out_data, out_data,
                                                 rows, cols,
                                                 {carg.out_node});

        g.add_compiled(this, {out_node}, {out_data}, out_node, out_shape, out_data);
    }
    
}
