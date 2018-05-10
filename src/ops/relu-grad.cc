#include "relu-grad.hh"
#include "graph.hh"
#include "../runtime/node.hh"
#include "../memory/alloc.hh"

namespace ops
{

    ReluGrad::ReluGrad(Op* z, Op* dout)
        : Op(z->shape_get(), {z, dout})
    {}

    void ReluGrad::compile()
    {
        auto& g = Graph::instance();

        auto& cz_out = g.compiled(preds()[0]);
        auto& cdout = g.compiled(preds()[1]);

        std::size_t len = cz_out.out_shape.total();
        Shape out_shape = cz_out.out_shape;
        dbl_t* out_data = tensor_alloc(len);

        auto out_node = rt::Node::op_relu_grad(cz_out.out_data, cdout.out_data, out_data,
                                                  len,
                                                  {cz_out.out_node, cdout.out_node});

        g.add_compiled(this, {out_node}, {out_data}, out_node, out_shape, out_data);
    }
    
}
