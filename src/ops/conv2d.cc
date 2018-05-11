#include "conv2d.hh"
#include "graph.hh"
#include "../runtime/node.hh"
#include "../memory/alloc.hh"
#include "conv2d-input-grad.hh"
#include "conv2d-kernel-grad.hh"
#include "ops-builder.hh"
#include <cassert>
#include <stdexcept>
#include <math.h>

namespace ops
{

    Conv2D::Conv2D(Op* input, Op* kernel, const int strides[])
        : Op("conv2d",
            Shape({input->shape_get()[0],
                   (int)std::ceil(static_cast<float>(input->shape_get()[1]) / (float)strides[0]),
                   (int)std::ceil(static_cast<float>(input->shape_get()[2]) / (float)strides[1]),
                   kernel->shape_get()[3]}),
             {input, kernel})
        , m_strides(strides)
        , m_input_shape(input->shape_get())
        , m_kernel_shape(kernel->shape_get())
    {
      int in_height = input->shape_get()[1];
      int in_width = input->shape_get()[2];
      int filter_height = kernel->shape_get()[0];
      int filter_width = kernel->shape_get()[1];
      int pad_along_height = 0;
      int pad_along_width = 0;

      if (in_height % strides[0] == 0)
          pad_along_height = std::max(filter_height - strides[0], 0);
      else
          pad_along_height = std::max(filter_height - (in_height % strides[0]), 0);
      if (in_width % strides[1] == 0)
          pad_along_width = std::max(filter_width - strides[1], 0);
      else
          pad_along_width = std::max(filter_width - (in_width % strides[1]), 0);

      m_pad_top = pad_along_height / 2;
      m_pad_left = pad_along_width / 2;
    }

    void Conv2D::compile()
    {
        auto& g = Graph::instance();
        auto& cinput =  g.compiled(preds()[0]);
        auto& ckernel =  g.compiled(preds()[1]);

        std::size_t b = cinput.out_shape[0];
        std::size_t i = (std::size_t)std::ceil(
                          static_cast<float>(cinput.out_shape[1])
                          / (float)m_strides[0]);
        std::size_t j = (std::size_t)std::ceil(
                          static_cast<float>(cinput.out_shape[2])
                          / (float)m_strides[1]);
        std::size_t k = ckernel.out_shape[3];

        Shape out_shape({int(b), int(i), int(j), int(k)});
        dbl_t* out_data = tensor_alloc(out_shape.total());

        int input_size[4] = { cinput.out_shape[0], cinput.out_shape[1],
                              cinput.out_shape[2], cinput.out_shape[3]};

        int kernel_size[4] = { ckernel.out_shape[0], ckernel.out_shape[1],
                               ckernel.out_shape[2], ckernel.out_shape[3]};

        auto out_node = rt::Node::op_conv2d(cinput.out_data, ckernel.out_data,
                                            m_strides, m_pad_top, m_pad_left,
                                            out_data, input_size, kernel_size,
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
