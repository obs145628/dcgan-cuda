#include "conv2d.hh"
#include "graph.hh"
#include "../runtime/graph.hh"
#include "../runtime/node.hh"
#include "../memory/alloc.hh"

namespace ops
{

	Conv2D::Conv2D(Op* input, Op* kernel, const int* strides)
	: Op(Shape({input->shape_get()[0],
							(input->shape_get()[1] - kernel->shape_get()[0]) / strides[0] + 1,
							(input->shape_get()[2] - kernel->shape_get()[1]) / strides[1] + 1,
							kernel->shape_get()[3]}),
		 {input, kernel})
		,m_strides(strides)
	{}

	void Conv2D::compile()
	{
		auto& g = Graph::instance();
		auto& cinput =  g.compiled(preds()[0]);
		auto& ckernel =  g.compiled(preds()[1]);

		std::size_t b = cinput.out_shape[0];
		std::size_t i = (cinput.out_shape[1] - ckernel.out_shape[0]) / m_strides[0] + 1;
		std::size_t j = (cinput.out_shape[2] - ckernel.out_shape[1]) / m_strides[1] + 1;
		std::size_t k = ckernel.out_shape[3];

		Shape out_shape({int(b), int(i), int(j), int(k)});
		dbl_t* out_data = tensor_alloc(out_shape.total());

		const int input_size[4] = {cinput.out_shape[0], cinput.out_shape[1],
																				cinput.out_shape[2], cinput.out_shape[3]};

		const int kernel_size[4] = {ckernel.out_shape[0], ckernel.out_shape[1],
																				ckernel.out_shape[2], ckernel.out_shape[3]};

		auto out_node = rt::Node::op_conv2d(cinput.out_data, ckernel.out_data, m_strides, out_data,
																				input_size, kernel_size, {cinput.out_node, ckernel.out_node});

		g.add_compiled(this, {out_node}, {out_data}, out_node, out_shape, out_data);
	}
}
