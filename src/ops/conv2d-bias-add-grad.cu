#include "conv2d-bias-add-grad.hh"
#include "graph.hh"
#include "../runtime/graph.hh"
#include "../runtime/node.hh"
#include "../memory/alloc.hh"

namespace ops
{

    Conv2DBiasAddGrad::Conv2DBiasAddGrad(Op* dz)
        : Op("conv2d_bias_add_grad",
              Shape({dz->shape_get()[0], dz->shape_get()[1],
                    dz->shape_get()[2], dz->shape_get()[3]}),
             {dz})
    {}

    void Conv2DBiasAddGrad::compile()
    {
        auto& g = Graph::instance();
        auto& cz =  g.compiled(preds()[0]);

        Shape out_shape({cz.out_shape[3]});
        dbl_t* out_data = tensor_alloc(out_shape.total());

        int size[4] = {
            cz.out_shape[0], cz.out_shape[1],
            cz.out_shape[2], cz.out_shape[3]
        };

        auto out_node = rt::Node::op_mat_sum_cols(cz.out_data, out_data,
                                                  size[0] * size[1] * size[2], size[3],
                                                  {cz.out_node});
        
        //auto out_node = rt::Node::op_conv2d_bias_add_grad(cz.out_data, size, out_data,
        //{cz.out_node});

        g.add_compiled(this, {out_node}, {out_data}, out_node, out_shape, out_data);
    }
}
