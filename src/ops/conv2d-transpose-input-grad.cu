#include "conv2d-transpose-input-grad.hh"
#include "graph.hh"
#include "../runtime/graph.hh"
#include "../runtime/node.hh"
#include "../memory/alloc.hh"

namespace ops
{

    Conv2DTransposeInputGrad::Conv2DTransposeInputGrad(Op* y, Op* kernel,
                                    const int strides[], const int input_size[])
        : Op("conv2d_transpose_input_grad",
            Shape({input_size[0], input_size[1],
                   input_size[2], input_size[3]}),
             {y, kernel})
        , m_strides(strides)
    {
        m_input_size[0] = input_size[0];
        m_input_size[1] = input_size[1];
        m_input_size[2] = input_size[2];
        m_input_size[3] = input_size[3];
    }

    void Conv2DTransposeInputGrad::compile()
    {
        auto& g = Graph::instance();
        auto& cy =  g.compiled(preds()[0]);
        auto& ckernel =  g.compiled(preds()[1]);

        Shape out_shape({m_input_size[0], m_input_size[1], m_input_size[2], m_input_size[3]});
        dbl_t* out_data = tensor_alloc(out_shape.total());

        int y_size[4] = { cy.out_shape[0], cy.out_shape[1],
                          cy.out_shape[2], cy.out_shape[3]};

        int kernel_size[4] = { ckernel.out_shape[0], ckernel.out_shape[1],
                               ckernel.out_shape[2], ckernel.out_shape[3]};

        auto out_node = rt::Node::op_conv2d_transpose_input_grad(cy.out_data, ckernel.out_data,
                                            m_strides, out_data, y_size,
                                            kernel_size, m_input_size,
                                            {cy.out_node, ckernel.out_node});

        g.add_compiled(this, {out_node}, {out_data}, out_node, out_shape, out_data);
    }
}
