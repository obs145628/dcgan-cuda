#include "vect-sigmoid.hh"
#include <cassert>
#include <stdexcept>
#include "graph.hh"
#include "ops-builder.hh"
#include "sigmoid-grad.hh"
#include "../runtime/node.hh"
#include "../memory/alloc.hh"

namespace ops
{

    VectSigmoid::VectSigmoid(Op* arg)
        : Op("vect-sigmoid", arg->shape_get(), {arg})
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


    Op* VectSigmoid::child_grad(std::size_t index, Op* dout)
    {
        assert(index < 1);
        (void) index;

        if (dout == nullptr)
            throw std::runtime_error {"grad(Sigmoid) can't be computed on last node"};

        auto& builder = OpsBuilder::instance();
        return builder.sigmoid_grad(this, dout);
    }
}
