#include "op.hh"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include "graph.hh"

namespace ops
{

    namespace
    {

        std::string unique_name(std::string str)
        {
            auto id = Graph::instance().ops_by_name().size();
            return str + ":" + std::to_string(id);
        }
        
    }

    Op::Op(const std::string& name,
           const Shape& shape,
           const std::vector<Op*> preds,
           const std::vector<Op*> succs)
        : name_(unique_name(name))
        , shape_(shape)
        , preds_(preds)
        , succs_(succs)
    {
        for (auto pred : preds_)
            pred->succs_.push_back(this);
    }

    const std::string& Op::name_get()
    {
        return name_;
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
