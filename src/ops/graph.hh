#pragma once

#include <map>
#include <set>
#include <vector>
#include "fwd.hh"
#include "shape.hh"
#include "../memory/types.hh"
#include "../runtime/graph.hh"

namespace ops
{

    class Graph
    {
    public:

        static Graph& instance();

        Graph(const Graph&) = delete;
        Graph(Graph&&) = delete;
        Graph& operator=(const Graph&) = delete;
        Graph& operator=(Graph&&) = delete;
        ~Graph();

        const std::vector<Op*>& ops_list() const;
        const std::map<Input*, Shape>& input_shapes_get();

        void add(Op* op);

        void compile(const std::map<Input*, Shape>& inputs);

        void run(std::vector<Op*> ops,
                 const std::map<Input*, std::pair<const dbl_t*, Shape>>& inputs = {},
                 const std::vector<dbl_t*>& outputs = {});

        /**
         * Add the compiled version of a node
         * op - op to be compiled
         * nodes - nodes created to compute this operation
         *   (somes op doesn't need any)
         * tensors - tensors to be free when this node must be recompiled
         *   (it happens when the dimesion changes)
         * out_node - the final node of the op, might be nullptr if no node
         * out_shape - the real shape of the node (dimensions are resolved)
         * out_data - pointer to the output tensor (this one is not null)
         */
        void add_compiled(Op* op, const std::vector<rt::Node*> nodes,
                          std::vector<dbl_t*> tensors,
                          rt::Node* out_node, const Shape& out_shape, dbl_t* out_data);

        const CompiledOp& compiled(Op* op);

        Op* gradient(Op* out, Op* var);
        

        
    private:
        std::vector<Op*> ops_;
        std::map<Input*, Shape> input_shapes_;

        rt::Graph full_rt_graph_;
        std::map<Op*, CompiledOp> compiled_ops_;

        std::map<std::pair<Op*, Op*>, Op*> grads_;

        void remove_compiled_rec_(Op* op);

        Graph();

        void compile_(Op* node);

        Op* compute_gradient_(Op* out, Op* var);
    };
    
}
