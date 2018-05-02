#pragma once

#include "shape.hh"
#include "fwd.hh"
#include "../runtime/fwd.hh"
#include "../memory/types.hh"
#include "../memory/alloc.hh"

namespace ops
{

    struct CompiledOp
    {
	CompiledOp(Op* op, const std::vector<rt::Node*> nodes, std::vector<dbl_t*> tensors,
		   rt::Node* out_node, const Shape& out_shape, dbl_t* out_data)
	    : op(op)
	    , nodes(nodes)
	    , tensors(tensors)
	    , out_node(out_node)
	    , out_shape(out_shape)
	    , out_data(out_data)
	    {}

	CompiledOp(const CompiledOp&) = delete;

	CompiledOp(CompiledOp&& cop)
	    : op(cop.op)
	    , nodes(cop.nodes)
	    , tensors(cop.tensors)
	    , out_node(cop.out_node)
	    , out_shape(cop.out_shape)
	    , out_data(cop.out_data)
	    {
		cop.tensors.clear();
	    }

	CompiledOp& operator=(const CompiledOp&) = delete;
	CompiledOp& operator==(CompiledOp&&) = delete;

	~CompiledOp()
	{
	    for (auto t: tensors)
		tensor_free(t);
	}
        

	Op* op;
	std::vector<rt::Node*> nodes;
	std::vector<dbl_t*> tensors;
	rt::Node* out_node;
	Shape out_shape;
	dbl_t* out_data;
    };
    
}
