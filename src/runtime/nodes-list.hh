#pragma once

#include <iostream>
#include <vector>
#include "fwd.hh"

namespace rt
{

    class NodesList
    {
    public:

        NodesList(const std::vector<Node*> nodes);

        std::size_t size() const;
        const std::vector<Node*>& nodes() const;
        const std::vector<std::vector<std::size_t>>& preds() const;

    private:
        std::vector<Node*> nodes_;
        std::vector<std::vector<std::size_t>> preds_;
        
    };

    std::ostream& operator<<(std::ostream& os, const NodesList& list);
}
