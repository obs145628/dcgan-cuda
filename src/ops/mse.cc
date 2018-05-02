#include "mse.hh"
#include "graph.hh"
#include "../runtime/node.hh"
#include "../memory/alloc.hh"

namespace ops
{

    MSE::MSE(Op* y, Op* y_hat)
	: Op(Shape{}, {y, y_hat})
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
    
}
