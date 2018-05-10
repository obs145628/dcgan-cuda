#include "dot-graph.hh"
#include <fstream>

namespace utils
{

    DotGraph::DotGraph(const std::string& name)
        : name_(name)
    {}

    void DotGraph::write_dot(std::ostream& os) const
    {
        os << "digraph " << name_ << "\n";
        os << "{\n";


        for (auto it : vs_)
        {
            os << "  n" << it.second << " [label=\"";
            for (char c : it.first)
            {
                if (c == '"')
                    os << "\\\"";
                else
                    os << c;
            }
            os << "\"]\n";
        }

        for (auto it : es_)
            os << "  n" << it.first << " -> n" << it.second << "\n";

        os << "}\n";
    }
    
    void DotGraph::write_file(const std::string& path) const
    {
        std::ofstream os(path);
        write_dot(os);
    }

    void DotGraph::write_png(const std::string& path) const
    {
        (void) path;
    }

    std::size_t DotGraph::add_vertex(const std::string& name)
    {
        auto it = vs_.find(name);
        if (it != vs_.end())
            return it->second;
        
        std::size_t id = vs_.size();
        vs_[name] = id;
        return id;
    }

    void DotGraph::add_edge(const std::string& a, const std::string& b)
    {
        auto v1 = add_vertex(a);
        auto v2 = add_vertex(b);
        es_.insert({v1, v2});
    }

}
