#pragma once

#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace utils
{

    class DotGraph
    {
    public:

        DotGraph(const std::string& name = "G");

        void write_dot(std::ostream& os) const;
        void write_file(const std::string& path) const;
        void write_png(const std::string& path) const;

        std::size_t add_vertex(const std::string& name);
        void add_edge(const std::string& a, const std::string& b);

    private:
        std::string name_;
        std::map<std::string, std::size_t> vs_;
        std::set<std::pair<std::size_t, std::size_t>> es_;
    };

}
