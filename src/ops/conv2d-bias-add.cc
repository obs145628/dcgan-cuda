#include "conv2d-bias-add.hh"
#include "graph.hh"
#include "../runtime/graph.hh"
#include "../runtime/node.hh"
#include "../memory/alloc.hh"

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

        auto out_node = rt::Node::op_conv2d_bias_add(cz.out_data, cbias.out_data,
                                                     out_data, input_size,
                                                    {cz.out_node, cbias.out_node});

        g.add_compiled(this, {out_node}, {out_data}, out_node, out_shape, out_data);
    }
}
