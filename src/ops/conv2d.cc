#include "conv2d.hh"
#include "graph.hh"
#include "../runtime/graph.hh"
#include "../runtime/node.hh"
#include "../memory/alloc.hh"

namespace ops
{
	
	Conv2D::Conv2D(Op* input, Op* mask)
	: Op(Shape({input->shape_get()[0], input->shape_get()[1]}),
		 {input, mask})
	{}
	
	void Conv2D::compile()
	{
		auto& g = Graph::instance();
		auto& cinput =  g.compiled(preds()[0]);
		auto& cmask =  g.compiled(preds()[1]);
		
		std::size_t n = cinput.out_shape[0];
		std::size_t p = cinput.out_shape[1];
		
		Shape out_shape({int(n), int(p)});
		dbl_t* out_data = tensor_alloc(out_shape.total());
		
		auto out_node = rt::Node::op_conv2d(cinput.out_data,
	cmask.out_data, out_data, n, p, {cinput.out_node, cmask.out_node});
	
		g.add_compiled(this, {out_node}, {out_data}, out_node, out_shape,
						out_data);
	}
}