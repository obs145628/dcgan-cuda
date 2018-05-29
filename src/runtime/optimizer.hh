#pragma once

#include "fwd.hh"
#include <map>
#include <vector>

namespace rt
{

    /**
     * Create a completetely new and independant graph
     * The original graph is not modified
     * The returned graph must be deleted
     *
     * Transform every operations into multiple operations
     * Convert to SIMD operations when possible
     */
    Graph* optimize(const Graph& graph, std::map<Node*, Node*>& optis);


    /**
     * Convert a list of nodes of a non optimized graph 
     * to the list of nodes of the optimized graph
     */
    std::vector<Node*> convert_nodes(const std::vector<Node*> nodes,
                                     const std::map<Node*, Node*>& optis);
    
}
