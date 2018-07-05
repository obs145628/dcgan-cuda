#pragma once

#include <string>
#include <vector>
#include "../memory/types.hh"


namespace celeba
{

    dbl_t* load(const std::vector<std::size_t>& idxs);

    dbl_t* load(std::size_t idx_beg, std::size_t idx_end);

    void save_samples(const dbl_t* data, std::size_t width, std::size_t height,
                      const std::string& path);
    
    
}
