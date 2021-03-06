#include "vect-relu-leaky.hh"
#include <cassert>
#include <stdexcept>
#include "ops-builder.hh"
#include "leaky-relu-grad.hh"
#include "graph.hh"
#include "../runtime/node.hh"
#include "../memory/alloc.hh"

namespace ops
{

    VectReluLeaky::VectReluLeaky(Op* arg, const dbl_t alpha)
        : Op("vect_relu_leaky", arg->shape_get(), {arg})
        , alpha(alpha)
    {}

    void VectReluLeaky::compile()
    {
        auto& g = Graph::instance();
        auto& carg = g.compiled(preds()[0]);


        std::size_t len = carg.out_shape.total();
        Shape out_shape = carg.out_shape;
        dbl_t* out_data = tensor_alloc(len);

        auto out_node = rt::Node::op_relu_leaky(carg.out_data, out_data,
                                                len, alpha,
                                                {carg.out_node});

        g.add_compiled(this, {out_node}, {out_data}, out_node, out_shape, out_data);
    }

    Op* VectReluLeaky::child_grad(std::size_t index, Op* dout)
    {
        assert(index < 1);
        (void) index;

        if (dout == nullptr)
            throw std::runtime_error {"grad(LeakyRelu) can't be computed on last node"};

        auto& builder = OpsBuilder::instance();
        return builder.leaky_relu_grad(preds()[0], dout, alpha);
    }
}
