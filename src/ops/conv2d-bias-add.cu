#include "conv2d-bias-add.hh"
#include "conv2d-bias-add-grad.hh"
#include "graph.hh"
#include "../runtime/graph.hh"
#include "../runtime/node.hh"
#include "../memory/alloc.hh"
#include "ops-builder.hh"
#include <cassert>
#include <stdexcept>

namespace ops
{

    Conv2DBiasAdd::Conv2DBiasAdd(Op* z, Op* bias)
        : Op("conv2d_bias_add",
             Shape({z->shape_get()[0], z->shape_get()[1],
                    z->shape_get()[2], z->shape_get()[3]}),
             {z, bias})
    {}

    void Conv2DBiasAdd::compile()
    {
        auto& g = Graph::instance();
        auto& cz =  g.compiled(preds()[0]);
        auto& cbias =  g.compiled(preds()[1]);

        Shape out_shape({int(cz.out_shape[0]), int(cz.out_shape[1]),
                         int(cz.out_shape[2]), int(cz.out_shape[3])});
        dbl_t* out_data = tensor_alloc(out_shape.total());

        int input_size[4] = { cz.out_shape[0], cz.out_shape[1],
                              cz.out_shape[2], cz.out_shape[3]};


        auto out_node = rt::Node::op_mat_rvect_add(cz.out_data, cbias.out_data, out_data,
                                                   input_size[0] * input_size[1] * input_size[2],
                                                   input_size[3],
                                                   {cz.out_node, cbias.out_node});
        
        /*
        auto out_node = rt::Node::op_conv2d_bias_add(cz.out_data, cbias.out_data,
                                                     out_data, input_size,
                                                    {cz.out_node, cbias.out_node});
        */

        g.add_compiled(this, {out_node}, {out_data}, out_node, out_shape, out_data);
    }

    Op* Conv2DBiasAdd::child_grad(std::size_t index, Op* dout)
    {
        assert(index < 2);
        if (dout == nullptr)
            throw std::runtime_error {"conv2d_bias_add must not be the final node of the gradient"};

        auto& builder = OpsBuilder::instance();

        if (index == 0)
            return dout;
        else
            return builder.conv2d_bias_add_grad(dout);
    }
}
