#include "conv2d-kernel-grad.hh"
#include "graph.hh"
#include "../runtime/graph.hh"
#include "../runtime/node.hh"
#include "../memory/alloc.hh"

namespace ops
{

    Conv2DKernelGrad::Conv2DKernelGrad(Op* y, Op* input, const int strides[], const int kernel_size[])
        : Op("conv2d_kernel_grad",
            Shape({kernel_size[0], kernel_size[1],
                   kernel_size[2],kernel_size[3]}),
             {y, input})
        , m_strides(strides)
        , m_kernel_size(kernel_size)
    {}

    void Conv2DKernelGrad::compile()
    {
        auto& g = Graph::instance();
        auto& cy =  g.compiled(preds()[0]);
        auto& cinput =  g.compiled(preds()[1]);

        Shape out_shape({m_kernel_size[0], m_kernel_size[1], m_kernel_size[2], m_kernel_size[3]});
        dbl_t* out_data = tensor_alloc(out_shape.total());

        int y_size[4] = { cy.out_shape[0], cy.out_shape[1],
                          cy.out_shape[2], cy.out_shape[3]};

        int input_size[4] = { cinput.out_shape[0], cinput.out_shape[1],
                              cinput.out_shape[2], cinput.out_shape[3]};

        auto out_node = rt::Node::op_conv2d_kernel_grad(cy.out_data, cinput.out_data,
                                            m_strides, out_data, y_size,
                                            input_size,
                                            {cy.out_node, cinput.out_node});

        g.add_compiled(this, {out_node}, {out_data}, out_node, out_shape, out_data);
    }
}