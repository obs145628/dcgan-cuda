#include "conv2d.hh"
#include "graph.hh"
#include "../runtime/node.hh"
#include "../memory/alloc.hh"
#include "conv2d-input-grad.hh"
#include "conv2d-kernel-grad.hh"
#include "ops-builder.hh"
#include <cassert>
#include <stdexcept>

namespace ops
{

    Conv2D::Conv2D(Op* input, Op* kernel, const int strides[])
        : Op("conv2d",
            Shape({input->shape_get()[0],
                   (input->shape_get()[1] - kernel->shape_get()[0]) / strides[0] + 1,
                   (input->shape_get()[2] - kernel->shape_get()[1]) / strides[1] + 1,
                   kernel->shape_get()[3]}),
             {input, kernel})
        , m_strides(strides)
        , m_input_shape(input->shape_get())
        , m_kernel_shape(kernel->shape_get())
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

        int input_size[4] = { cinput.out_shape[0], cinput.out_shape[1],
                              cinput.out_shape[2], cinput.out_shape[3]};

        int kernel_size[4] = { ckernel.out_shape[0], ckernel.out_shape[1],
                               ckernel.out_shape[2], ckernel.out_shape[3]};

        auto out_node = rt::Node::op_conv2d(cinput.out_data, ckernel.out_data,
                                            m_strides, out_data, input_size,
                                            kernel_size,
                                            {cinput.out_node, ckernel.out_node});

        g.add_compiled(this, {out_node}, {out_data}, out_node, out_shape, out_data);
    }

    Op* Conv2D::child_grad(std::size_t index, Op* dout)
    {
        assert(index < 2);
        if (dout == nullptr)
            throw std::runtime_error {"conv2d must not be the final node of the gradient"};

        auto& builder = OpsBuilder::instance();

        int input_size[4] = { m_input_shape[0], m_input_shape[1],
                              m_input_shape[2], m_input_shape[3]};

        int kernel_size[4] = { m_kernel_shape[0], m_kernel_shape[1],
                               m_kernel_shape[2], m_kernel_shape[3]};
        if (index == 0)
          return builder.conv2d_input_grad(dout , preds()[1], m_strides, input_size);
        else
          return builder.conv2d_kernel_grad(dout, preds()[0], m_strides, kernel_size);
    }
}
