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

	std::vector<Node*> topological_sort(const std::vector<Node*>& vals);

    private:
	std::vector<Node*> nodes_;

    };

}
