#include "softmax-cross-entropy.hh"
#include "graph.hh"
#include "../runtime/node.hh"
#include "../memory/alloc.hh"

namespace ops
{

    SoftmaxCrossEntropy::SoftmaxCrossEntropy(Op* y, Op* logits)
        : Op(Shape{}, {y, logits})
    {}

    void SoftmaxCrossEntropy::compile()
    {
        auto& g = Graph::instance();

        auto& cy = g.compiled(preds()[0]);
        auto& clogits = g.compiled(preds()[1]);

        std::size_t rows = cy.out_shape[0];
        std::size_t cols = cy.out_shape[1];
        Shape out_shape {};
        dbl_t* out_data = tensor_alloc(1);

        auto out_node = rt::Node::op_softmax_cross_entropy(cy.out_data, clogits.out_data, out_data,
                                                           rows, cols,
                                                           {cy.out_node, clogits.out_node});

        g.add_compiled(this, {out_node}, {out_data}, out_node, out_shape, out_data);
    }
    
}

