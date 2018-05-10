#include "sigmoid-cross-entropy-grad.hh"
#include "graph.hh"
#include "../runtime/node.hh"
#include "../memory/alloc.hh"

namespace ops
{

    SigmoidCrossEntropyGrad::SigmoidCrossEntropyGrad(Op* y, Op* logits)
        : Op("sigmoid_cross_entropy_grad", y->shape_get(), {y, logits})
    {}

    void SigmoidCrossEntropyGrad::compile()
    {
        auto& g = Graph::instance();

        auto& cy = g.compiled(preds()[0]);
        auto& clogits = g.compiled(preds()[1]);

        std::size_t len = cy.out_shape.total();
        Shape out_shape = cy.out_shape;
        dbl_t* out_data = tensor_alloc(len);

        auto out_node = rt::Node::op_sigmoid_cross_entropy_grad(cy.out_data, clogits.out_data, out_data,
                                                                len,
                                                                {cy.out_node, clogits.out_node});

        g.add_compiled(this, {out_node}, {out_data}, out_node, out_shape, out_data);
    }
    
}

