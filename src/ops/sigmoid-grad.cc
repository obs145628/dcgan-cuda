#include "sigmoid-grad.hh"
#include "graph.hh"
#include "../runtime/node.hh"
#include "../memory/alloc.hh"

namespace ops
{

    SigmoidGrad::SigmoidGrad(Op* sig_out, Op* dout)
        : Op("sigmoid_grad", sig_out->shape_get(), {sig_out, dout})
    {}

    void SigmoidGrad::compile()
    {
        auto& g = Graph::instance();

        auto& csig_out = g.compiled(preds()[0]);
        auto& cdout = g.compiled(preds()[1]);

        std::size_t len = csig_out.out_shape.total();
        Shape out_shape = csig_out.out_shape;
        dbl_t* out_data = tensor_alloc(len);

        auto out_node = rt::Node::op_sigmoid_grad(csig_out.out_data, cdout.out_data, out_data,
                                                  len,
                                                  {csig_out.out_node, cdout.out_node});

        g.add_compiled(this, {out_node}, {out_data}, out_node, out_shape, out_data);
    }
    
}
