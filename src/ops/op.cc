#include "op.hh"

#include <iostream>

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
    
}
