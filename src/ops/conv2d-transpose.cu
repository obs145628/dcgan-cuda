#include "conv2d-transpose.hh"
#include "graph.hh"
#include "../runtime/node.hh"
#include "../memory/alloc.hh"
#include "ops-builder.hh"
#include "conv2d-transpose-input-grad.hh"
#include "conv2d-transpose-kernel-grad.hh"
#include <cassert>
#include <stdexcept>
#include <cmath>

namespace ops
{

    Conv2DTranspose::Conv2DTranspose(Op* input, Op* kernel, const int out_size[],
                                     const int strides[])
        : Op("conv2d_transpose",
            Shape({out_size[0], out_size[1], out_size[2], out_size[3]}),
             {input, kernel}
             )
        , m_out_size(out_size)
        , m_strides(strides)
        , m_input_shape(input->shape_get())
        , m_kernel_shape(kernel->shape_get())
    {}

    void Conv2DTranspose::compile()
    {
        auto& g = Graph::instance();
        auto& cinput =  g.compiled(preds()[0]);
        auto& ckernel =  g.compiled(preds()[1]);

        Shape out_shape({m_out_size[0], m_out_size[1],
                        m_out_size[2], m_out_size[3]});
        dbl_t* out_data = tensor_alloc(out_shape.total());

        int input_size[4] = { cinput.out_shape[0], cinput.out_shape[1],
                              cinput.out_shape[2], cinput.out_shape[3]};

        int kernel_size[4] = { ckernel.out_shape[0], ckernel.out_shape[1],
                               ckernel.out_shape[2], ckernel.out_shape[3]};

        auto out_node = rt::Node::op_conv2d_transpose(cinput.out_data, ckernel.out_data,
                                            m_out_size, m_strides, out_data, input_size, kernel_size,
                                            {cinput.out_node, ckernel.out_node});

        g.add_compiled(this, {out_node}, {out_data}, out_node, out_shape, out_data);
    }

    Op* Conv2DTranspose::child_grad(std::size_t index, Op* dout)
    {
        assert(index < 2);
        if (dout == nullptr)
            throw std::runtime_error {"conv2d_transpose must not be the final node of the gradient"};

        auto& builder = OpsBuilder::instance();

        int input_size[4] = { m_input_shape[0], m_input_shape[1],
                              m_input_shape[2], m_input_shape[3]};

        int kernel_size[4] = { m_kernel_shape[0], m_kernel_shape[1],
                               m_kernel_shape[2], m_kernel_shape[3]};
        if (index == 0)
          return builder.conv2d_transpose_input_grad(dout, preds()[1], m_strides, input_size);
        else
          return builder.conv2d_transpose_kernel_grad(dout, preds()[0], m_strides, kernel_size);
    }
}
