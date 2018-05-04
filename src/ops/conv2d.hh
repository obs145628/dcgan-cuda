#pragma once

#include "op.hh"

namespace ops
{

	class Conv2D : public Op
	{
	public:
	Conv2D(Op* input, Op* mask);
	
	virtual void compile() override;
	}
}