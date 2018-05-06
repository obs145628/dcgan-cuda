#pragma once

#include "shape.hh"
#include <string>

#include "compiled-op.hh"

namespace ops
{

    class Op
    {

    public:
	Op(const Shape& shape,
	   const std::vector<Op*> preds = {},
	   const std::vector<Op*> succs = {});
	virtual ~Op() = default;

	const Shape& shape_get() const;
	std::vector<Op*> preds();
	std::vector<Op*> succs();

	virtual void compile() = 0;

    private:
	Shape shape_;
	std::vector<Op*> preds_;
	std::vector<Op*> succs_;
    };

}
