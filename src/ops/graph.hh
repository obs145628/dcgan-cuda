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

        /**
         * Get the real shapes of allocated inputs
         */
        const std::map<Input*, Shape>& input_shapes_get();

        /**
         * Add a new node to the graph, never called directly
         * Called by ops-builder
         */
        void add(Op* op);

        /**
         * Realise network computations
         * ops - the list of operations to be realised
         * input - the data for each input
         * output - the lis of pointers where each node result is stored
         *  it must be on the same order as the list of operation
         *  the pointer can be nullptr if the result is not needed
         */
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

        /**
         * Get the compiled result of an alredy compiled node
         * It's must be called on already compiled nodes
         */
        const CompiledOp& compiled(Op* op);

        /**
         * Return an opeation to compute nabla(out) / nabla(var)
         * Create a new graph operation to compute it if it doesn't exists
         * Automatically create intermediary nodes to compute required gradients
         * Use backpropagation algorithm
         */
        Op* gradient(Op* out, Op* var);

        /**
         * Enable / diable debugging
         * Display the compiled list of operations before running it
         */
        void debug_set(bool debug);

        
    private:
        std::vector<Op*> ops_;
        std::map<Input*, Shape> input_shapes_;

        rt::Graph full_rt_graph_;
        std::map<Op*, CompiledOp> compiled_ops_;

        std::map<std::pair<Op*, Op*>, Op*> grads_;
        bool debug_;

        void remove_compiled_rec_(Op* op);

        Graph();

        void compile_(Op* node);

        Op* compute_gradient_(Op* out, Op* var);
    };
    
}
