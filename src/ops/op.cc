#include "op.hh"

#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace ops
{

    Op::Op(const Shape& shape,
           const std::vector<Op*> preds,
           const std::vector<Op*> succs)
        : shape_(shape)
        , preds_(preds)
        , succs_(succs)
    {
        for (auto pred : preds_)
            pred->succs_.push_back(this);
    }

    const Shape& Op::shape_get() const
    {
        return shape_;
    }
    
    std::vector<Op*> Op::preds()
    {
        return preds_;
    }
    
    std::vector<Op*> Op::succs()
    {
        return succs_;
    }

    std::size_t Op::pred_index(Op* pred)
    {
        auto it = std::find(preds_.begin(), preds_.end(), pred);
        if (it == preds_.end())
            return -1;
        else
            return it - preds_.begin();
    }

    Op* Op::pred_of(Op* node)
    {
        for (auto succ: succs_)
            if (succ == node
                || (succ->pred_of(node)))
                return succ;
        return nullptr;
    }

    Op* Op::child_grad(std::size_t, Op*)
    {
        throw std::runtime_error {"It's not possible to compute the gradient of this node"};
    }
    
}
