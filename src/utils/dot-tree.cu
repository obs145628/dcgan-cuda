#include "dot-tree.hh"
#include <fstream>

namespace utils
{

    DotTree::DotTree(const std::string& label,
                     const std::vector<DotTree>& nodes)
        : label_(label)
    {
        for (const auto& node : nodes)
            nodes_.push_back(new DotTree(node));
    }

    DotTree::DotTree(const DotTree& tree)
        : label_(tree.label_)
    {
        for (auto node : tree.nodes_)
            nodes_.push_back(new DotTree(*node));
    }

    DotTree::~DotTree()
    {
        for (auto node : nodes_)
            delete node;
    }

    DotTree& DotTree::operator=(const DotTree& tree)
    {
        for (auto node : nodes_)
            delete node;
        nodes_.clear();
        label_ = tree.label_;

        for (const auto& node : tree.nodes_)
            nodes_.push_back(new DotTree(*node));

        return *this;
    }

    void DotTree::write_dot(std::ostream& os) const
    {
        int id = 0;
        os << "digraph tree\n";
        os << "{\n";
        write_(os, id);
        os << "}\n";
    }

    void DotTree::write_file(const std::string& path) const
    {
        std::ofstream os(path);
        write_dot(os);
    }

    void DotTree::write_png(const std::string& path) const
    {
        (void) path;
        /*
        write_file(TMP_DOT);
        Command::exec("dot -Tpng " + std::string(TMP_DOT)
                      + " -o " + path);
        */
    }

    int DotTree::write_(std::ostream& os, int& id) const
    {
        int self = id++;

        os << "  n" << self << " [label=\"";
        for (char c : label_)
            if (c == '"')
                os << "\\\"";
            else
                os << c;

        os << "\"]\n";

        for (auto node : nodes_)
        {
            int child = node->write_(os, id);
            os << "  n" << self << " -> n" << child << "\n";
        }


        return self;
    }

}
