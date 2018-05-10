#include "graph.hh"
#include "node.hh"
#include <algorithm>
#include <cassert>
#include <map>
#include <set>
#include <string>

#include <iostream>

namespace rt
{

    Graph::~Graph()
    {
        for (auto n : nodes_)
            delete n;
    }

    void Graph::add(Node* node)
    {
        assert(node);
        nodes_.push_back(node);
    }

    void Graph::remove(Node* node)
    {
        assert(node);
        auto it = std::find(nodes_.begin(), nodes_.end(), node);
        assert(it != nodes_.end());
        nodes_.erase(it);

        for (auto x : node->preds)
        {
            auto pred_it = std::find(x->succs.begin(), x->succs.end(), node);
            assert(pred_it != x->succs.end());
            x->succs.erase(pred_it);
        }

        for (auto x : node->succs)
            remove(x);
        delete node;
    }

    const std::vector<Node*> Graph::nodes() const
    {
        return nodes_;
    }

    namespace
    {

        void add_preds(Node* node, std::set<Node*>& set)
        {
            if (!set.insert(node).second)
                return;
            for (auto x : node->preds)
                add_preds(x, set);
        }

        std::vector<Node*> get_preds(Node* node, std::set<Node*>& graph)
        {
            std::vector<Node*> res;
            for (auto x : node->preds)
                if (graph.find(x) != graph.end())
                    res.push_back(x);
            return res;
        }

        std::vector<Node*> get_succs(Node* node, std::set<Node*>& graph)
        {
            std::vector<Node*> res;
            for (auto x : node->succs)
                if (graph.find(x) != graph.end())
                    res.push_back(x);
            return res;
        }

    }

    /**
     * L ← Empty list that will contain the sorted elements
     * S ← Set of all nodes with no incoming edge
     * while S is non-empty do
     *   remove a node n from S
     *   add n to tail of L
     *   for each node m with an edge e from n to m do
     *     remove edge e from the graph
     *     if m has no other incoming edges then
     *       insert m into S
     * if graph has edges then
     *   return error (graph has at least one cycle)
     * else
     *   return L (a topologically sorted order)
     */

    std::vector<Node*> Graph::topological_sort(const std::vector<Node*>& vals)
    {
        std::vector<Node*> res;
        std::set<Node*> graph;
        for (auto n : vals)
            add_preds(n, graph);

        std::vector<Node*> s;
        for (auto x : graph)
            if (get_preds(x, graph).empty())
                s.push_back(x);

        while (!s.empty())
        {
            Node* next = s.back();
            s.pop_back();
            res.push_back(next);
            for (auto succ : get_succs(next, graph))
            {
                if (get_preds(succ, graph).size() == 1)
                    s.push_back(succ);
            }
            graph.erase(next);
        }

        if (!graph.empty())
            throw std::runtime_error {"Topological sort failed"};


        return res;
    }


    namespace
    {

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

    void Graph::print_nodes(std::ostream& os,
                            const std::vector<Node*>& vals)
    {

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
            for (std::size_t i = 0; i < node->preds.size(); ++i)
            {
                auto it = std::find(vals.begin(), vals.end(), node->preds[i]);
                auto pos = std::distance(vals.begin(), it);
                os << pos;
                if (i + 1 != node->preds.size())
                    os << ", ";
            }
            os << "]";

            os << std::endl;
        }
        
    }


    namespace
    {

        std::string op_name(const Node* node,
                            std::map<const Node*, std::string>& names)
        {
            auto it = names.find(node);
            if (it != names.end())
                return it->second;

            std::string res = Node::OP_NAMES[node->type] + std::string(":") + std::to_string(names.size());
            names[node] = res;
            return res;
        }
    }


    utils::DotGraph Graph::to_dot_graph() const
    {
        std::map<const Node*, std::string> names;
        utils::DotGraph g;
        for (auto n : nodes_)
            for (auto succ : n->succs)
                g.add_edge(op_name(n, names), op_name(succ, names));
        return g;
    }

}
