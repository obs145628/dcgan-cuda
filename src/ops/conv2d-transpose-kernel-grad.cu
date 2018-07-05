#include "conv2d-transpose-kernel-grad.hh"
#include "graph.hh"
#include "../runtime/graph.hh"
#include "../runtime/node.hh"
#include "../memory/alloc.hh"

namespace ops
{

    Conv2DTransposeKernelGrad::Conv2DTransposeKernelGrad(Op* y, Op* input,
                                    const int strides[], const int kernel_size[])
        : Op("conv2d_transpose_kernel_grad",
            Shape({kernel_size[0], kernel_size[1],
                   kernel_size[2],kernel_size[3]}),
             {y, input})
        , m_strides(strides)
    {
        m_kernel_size[0] = kernel_size[0];
        m_kernel_size[1] = kernel_size[1];
        m_kernel_size[2] = kernel_size[2];
        m_kernel_size[3] = kernel_size[3];
   }

    void Conv2DTransposeKernelGrad::compile()
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

        int kernel_size[4] = { shape_get()[0], shape_get()[1],
                               shape_get()[2], shape_get()[3]};
        
        auto out_node = rt::Node::op_conv2d_transpose_kernel_grad(cy.out_data, cinput.out_data,
                                            m_strides, out_data, y_size,
                                            input_size, kernel_size,
                                            {cy.out_node, cinput.out_node});

        g.add_compiled(this, {out_node}, {out_data}, out_node, out_shape, out_data);
    }
}
