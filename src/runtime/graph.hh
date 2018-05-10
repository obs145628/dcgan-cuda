#pragma once

#include <iostream>
#include <vector>
#include "fwd.hh"
#include "../utils/dot-graph.hh"

namespace rt
{

    /**
     * Runtime Graph
     * Holds maths operations, operands, and their dependencies
     */
    class Graph
    {

    public:

        Graph() = default;
        ~Graph();
        Graph(const Graph&) = delete;
        Graph(Graph&&) = delete;
        Graph& operator=(const Graph&) = delete;
        Graph& operator=(Graph&&) = delete;

        /**
         * Add a new node to the graph
         * Graph takes ownership of the node
         */
        void add(Node* node);

        /**
         * Remove a node to the graph
         * All successors are also removed
         */
        void remove(Node* node);

        const std::vector<Node*> nodes() const;

        /**
         * Applies a topological sort on a list of nodes
         * Returns a list with all the nodes that must be executed
         * List sorted in order of execution
         * Doesn't contains just the nodes of vals, but their predecessors recursively
         */
        std::vector<Node*> topological_sort(const std::vector<Node*>& vals);


        /**
         * print all nodes into an assembly instructions list style
         */
        static void print_nodes(std::ostream& os,
                                const std::vector<Node*>& vals);

        /**
         * Returns a new vector of nodes without nop
         */
        static std::vector<Node*> clear_nops(const std::vector<Node*>& nodes);

        /**
         * Transform the graph into dot-graph
         * for debugging purposes only
         */
        utils::DotGraph to_dot_graph() const;
        

    private:
        std::vector<Node*> nodes_;

    };

}
