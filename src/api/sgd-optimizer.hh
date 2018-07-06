#pragma once

#include <vector>
#include "../memory/types.hh"
#include "../ops/fwd.hh"

class SGDOptimizer
{

public:
    SGDOptimizer(dbl_t learning_rate);

    ops::Op* minimize(ops::Op* loss,
                      const std::vector<ops::Variable*>& vars = {});

private:
    dbl_t learning_rate_;
};
