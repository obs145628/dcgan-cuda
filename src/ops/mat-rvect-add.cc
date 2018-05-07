#include "mat-rvect-add.hh"
#include "graph.hh"
#include "../runtime/graph.hh"
#include "../runtime/node.hh"
#include "../memory/alloc.hh"

namespace ops
{

    MatRvectAdd::MatRvectAdd(Op* left, Op* right)
        : Op(left->shape_get(), {left, right})
    {}

    void MatRvectAdd::compile()
    {
        auto& g = Graph::instance();
        auto& cleft =  g.compiled(preds()[0]);
        auto& cright =  g.compiled(preds()[1]);

        std::size_t n = cleft.out_shape[0];
        std::size_t p = cleft.out_shape[1];
        Shape out_shape({int(n), int(p)});
        dbl_t* out_data = tensor_alloc(out_shape.total());

        auto out_node = rt::Node::op_mat_rvect_add(cleft.out_data, cright.out_data, out_data,
                                                   n, p, {cleft.out_node, cright.out_node});


        g.add_compiled(this, {out_node}, {out_data}, out_node, out_shape, out_data);
    }

}
