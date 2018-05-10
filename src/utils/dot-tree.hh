#pragma once

#include <iostream>
#include <string>
#include <vector>

namespace utils
{

    class DotTree
    {
    public:

        DotTree(const std::string& label = {},
                const std::vector<DotTree>& nodes = {});
        DotTree(const DotTree& tree);
        ~DotTree();

        DotTree& operator=(const DotTree& tree);

        void write_dot(std::ostream& os) const;

        void write_file(const std::string& path) const;
        void write_png(const std::string& path) const;

    private:

        std::string label_;
        std::vector<DotTree*> nodes_;

        int write_(std::ostream& os, int& id) const;
  
    };

}
