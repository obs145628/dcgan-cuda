#include "mat-mat-mul.hh"
#include <cassert>
#include "graph.hh"
#include "../runtime/graph.hh"
#include "../runtime/node.hh"
#include "../memory/alloc.hh"

namespace ops
{

    MatMatMul::MatMatMul(Op* left, Op* right, bool left_tr, bool right_tr)
        : Op("mat_mat_mul",
             Shape({left->shape_get()[left_tr], right->shape_get()[!right_tr]}),
             {left, right})
        , left_tr_(left_tr)
        , right_tr_(right_tr)
    {}

    void MatMatMul::compile()
    {
        auto& g = Graph::instance();
        auto& cleft =  g.compiled(preds()[0]);
        auto& cright =  g.compiled(preds()[1]);

        std::size_t n = cleft.out_shape[left_tr_];
        std::size_t p = cleft.out_shape[!left_tr_];
        std::size_t q = cright.out_shape[!right_tr_];
        Shape out_shape({int(n), int(q)});
        dbl_t* out_data = tensor_alloc(out_shape.total());

        rt::Node* out_node = nullptr;

        if (left_tr_ && !right_tr_)
            out_node = rt::Node::op_tmat_mat_mul(cleft.out_data, cright.out_data, out_data,
                                                 n, p, q,
                                                 {cleft.out_node, cright.out_node});

        else if (!left_tr_ && right_tr_)
            out_node = rt::Node::op_mat_tmat_mul(cleft.out_data, cright.out_data, out_data,
                                                 n, p, q,
                                                 {cleft.out_node, cright.out_node});

        else if (left_tr_ && right_tr_)
        {
            
        }

        else
            out_node = rt::Node::op_mat_mat_mul(cleft.out_data, cright.out_data, out_data,
                                                n, p, q,
                                                {cleft.out_node, cright.out_node});

        
        assert(out_node);
        g.add_compiled(this, {out_node}, {out_data}, out_node, out_shape, out_data);
    }
    
}
