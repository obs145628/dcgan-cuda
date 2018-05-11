#include "mse.hh"
#include <cassert>
#include <stdexcept>
#include "graph.hh"
#include "mse-grad.hh"
#include "ops-builder.hh"
#include "../runtime/node.hh"
#include "../memory/alloc.hh"

namespace ops
{

    MSE::MSE(Op* y, Op* y_hat)
        : Op("mse", Shape{}, {y, y_hat})
    {}

    void MSE::compile()
    {
        auto& g = Graph::instance();

        auto& cy = g.compiled(preds()[0]);
        auto& cy_hat = g.compiled(preds()[1]);

        std::size_t rows = cy.out_shape[0];
        std::size_t cols = cy.out_shape[1];
        Shape out_shape {};
        dbl_t* out_data = tensor_alloc(1);

        auto out_node = rt::Node::op_mse(cy.out_data, cy_hat.out_data, out_data,
                                         rows, cols,
                                         {cy.out_node, cy_hat.out_node});

        g.add_compiled(this, {out_node}, {out_data}, out_node, out_shape, out_data);
    }

    Op* MSE::child_grad(std::size_t index, Op* dout)
    {
        assert(index < 2);
        if (index == 0)
            throw std::runtime_error {"Can't compute gradient of MSE for y"};

        if (dout != nullptr)
            throw std::runtime_error {"MSE must be the final node of the gradient"};


        auto& builder = OpsBuilder::instance();
        return builder.mse_grad(preds()[0] , preds()[1]);
    }
    
}
