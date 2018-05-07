#include "vect-sigmoid.hh"
#include "graph.hh"
#include "../runtime/node.hh"
#include "../memory/alloc.hh"

namespace ops
{

    VectSigmoid::VectSigmoid(Op* arg)
        : Op(arg->shape_get(), {arg})
    {}

    void VectSigmoid::compile()
    {
        auto& g = Graph::instance();
        auto& carg = g.compiled(preds()[0]);

        std::size_t len = carg.out_shape.total();
        Shape out_shape = carg.out_shape;
        dbl_t* out_data = tensor_alloc(len);

        auto out_node = rt::Node::op_sigmoid(carg.out_data, out_data,
                                             len,
                                             {carg.out_node});

        g.add_compiled(this, {out_node}, {out_data}, out_node, out_shape, out_data);
    }
}
