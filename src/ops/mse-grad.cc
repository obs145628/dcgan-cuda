#include "mse-grad.hh"
#include "graph.hh"
#include "../runtime/node.hh"
#include "../memory/alloc.hh"

namespace ops
{

    MSEGrad::MSEGrad(Op* y, Op* y_hat)
        : Op(y->shape_get(), {y, y_hat})
    {}

    void MSEGrad::compile()
    {
        auto& g = Graph::instance();

        auto& cy = g.compiled(preds()[0]);
        auto& cy_hat = g.compiled(preds()[1]);

        std::size_t len = cy.out_shape.total();
        Shape out_shape = cy.out_shape;
        dbl_t* out_data = tensor_alloc(len);

        auto out_node = rt::Node::op_mse_grad(cy.out_data, cy_hat.out_data, out_data,
                                              len,
                                              {cy.out_node, cy_hat.out_node});

        g.add_compiled(this, {out_node}, {out_data}, out_node, out_shape, out_data);
    }
    
}
