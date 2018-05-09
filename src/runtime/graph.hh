#pragma once

#include <vector>
#include "fwd.hh"

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

    private:
        std::vector<Node*> nodes_;

    };

}
