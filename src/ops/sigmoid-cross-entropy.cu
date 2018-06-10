#include "sigmoid-cross-entropy.hh"
#include <cassert>
#include <stdexcept>
#include "graph.hh"
#include "ops-builder.hh"
#include "sigmoid-cross-entropy-grad.hh"
#include "../runtime/node.hh"
#include "../memory/alloc.hh"

namespace ops
{

    SigmoidCrossEntropy::SigmoidCrossEntropy(Op* y, Op* logits)
        : Op("sigmoid_cross_entropy", Shape{}, {y, logits})
    {}

    void SigmoidCrossEntropy::compile()
    {
        auto& g = Graph::instance();

        auto& cy = g.compiled(preds()[0]);
        auto& clogits = g.compiled(preds()[1]);

        std::size_t len = cy.out_shape.total();
        Shape out_shape {};
        dbl_t* out_data = tensor_alloc(1);

        auto out_node = rt::Node::op_sigmoid_cross_entropy(cy.out_data, clogits.out_data, out_data,
                                                           len,
                                                           {cy.out_node, clogits.out_node});

        g.add_compiled(this, {out_node}, {out_data}, out_node, out_shape, out_data);
    }

    Op* SigmoidCrossEntropy::child_grad(std::size_t index, Op* dout)
    {
        assert(index < 2);
        if (index == 0)
            throw std::runtime_error {"Can't compute gradient of sigmoid_cross_entropy for y"};

        if (dout != nullptr)
            throw std::runtime_error {"sigmoid_cross_entropy must be the final node of the gradient"};


        auto& builder = OpsBuilder::instance();
        return builder.sigmoid_cross_entropy_grad(preds()[0] , preds()[1]);
    }
    
}

