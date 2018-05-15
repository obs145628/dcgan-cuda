#include "nodes-list.hh"
#include "node.hh"
#include <algorithm>
#include <cassert>
#include <map>

namespace rt
{

    namespace
    {

        std::vector<Node*> preds_no_not(Node* n)
        {
            std::vector<Node*> res;
            for (auto pred : n->preds)
            {
                if (pred->type == Node::OP_NOP)
                {
                    auto rpreds = preds_no_not(pred);
                    res.insert(res.end(), rpreds.begin(), rpreds.end());
                }
                else
                    res.push_back(pred);
            }
            return res;
        }



        std::string tensor_name(const dbl_t* ptr,
                                std::map<const dbl_t*, std::string>& map)
        {
            auto it = map.find(ptr);
            if (it != map.end())
                return it->second;

            auto name = "t" + std::to_string(map.size());
            map[ptr] = name;
            return name;
        }
        
    }
        
    

    NodesList::NodesList(const std::vector<Node*> nodes)
    {
        for (auto n : nodes)
            if (n->type != Node::OP_NOP)
                nodes_.push_back(n);

        for (auto n : nodes_)
        {
            auto ppreds = preds_no_not(n);
            std::vector<std::size_t> npreds;
            for (auto p : ppreds)
            {
                auto it = std::find(nodes_.begin(), nodes_.end(), p);
                assert(it != nodes_.end());
                npreds.push_back(std::distance(nodes_.begin(), it));
            }
            preds_.push_back(npreds);
        }
    }

    std::size_t NodesList::size() const
    {
        return nodes_.size();
    }

    const std::vector<Node*>& NodesList::nodes() const
    {
        return nodes_;
    }
    
    const std::vector<std::vector<std::size_t>>& NodesList::preds() const
    {
        return preds_;
    }


    std::ostream& operator<<(std::ostream& os, const NodesList& list)
    {

        auto& vals = list.nodes();

        std::map<const dbl_t*, std::string> tmap;

        for (std::size_t i = 0; i < vals.size(); ++i)
        {
            auto node = vals[i];
            
            os << i << ": ";
            os << Node::OP_NAMES[node->type];

            os << " (" << tensor_name(node->out1, tmap);
            if (node->out2)
                os << ", " << tensor_name(node->out2, tmap);

            os << ") <= (" << tensor_name(node->in1, tmap);
            if (node->in2)
                os << ", " << tensor_name(node->in2, tmap) ;
            if (node->in3)
                os << ", " << tensor_name(node->in3, tmap);
            os << ") ";

            os << "{" << node->len1;
            if (node->len2)
                os << ", " << node->len2;
            if (node->len3)
                os << ", " << node->len3;
            os << "} ";
            
            os << "[";
            auto& preds = list.preds()[i];
            for (std::size_t i = 0; i < preds.size(); ++i)
            {
                os << preds[i];
                if (i + 1 != preds.size())
                    os << ", ";
            }
            os << "]";

            os << std::endl;
        }

        return os;
    }
    
}
