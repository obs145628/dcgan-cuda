#include "graph.hh"
#include "op.hh"
#include <cassert>
#include "../cpu/runner.hh"
#include "../memory/copy.hh"
#include "input.hh"

namespace ops
{

    Graph& Graph::instance()
    {
	static Graph graph;
	return graph;
    }

    Graph::Graph()
	: full_rt_graph_()
    {}

    Graph::~Graph()
    {
	for (auto x : ops_)
	    delete x;
    }

    const std::vector<Op*>& Graph::ops_list() const
    {
	return ops_;
    }

    void Graph::add(Op* op)
    {
	ops_.push_back(op);
    }

    const std::map<Input*, Shape>& Graph::input_shapes_get()
    {
	return input_shapes_;
    }

    void Graph::compile(const std::map<Input*, Shape>& inputs)
    {
	input_shapes_ = inputs;
	compiled_ops_.clear();
	for (auto node : ops_)
	    compile_(node);
    }

    void Graph::run(std::vector<Op*> ops,
		    const std::map<Input*, std::pair<const dbl_t*, Shape>>& inputs,
		    const std::vector<dbl_t*>& outputs)
    {

	//remove already compiled nodes with different shapes, and update shapes
	for (const auto& it: inputs)
	{
	    const auto& shape = it.second.second;

	    auto old_it = input_shapes_.find(it.first);
	    if (old_it != input_shapes_.end()
		&& old_it->second != shape)
	    {
		auto op_it = compiled_ops_.find(it.first);
		assert(op_it != compiled_ops_.end());
		for (auto x : op_it->second.nodes)
		    full_rt_graph_.remove(x);
		remove_compiled_rec_(it.first);
	    }

	    input_shapes_[it.first] = shape;
	}

	// compare nodes that are not already compiled and build ops list
	std::vector<rt::Node*> rt_ops;
	for (auto o : ops)
	{
	    auto it = compiled_ops_.find(o);
	    if (it == compiled_ops_.end())
	    {
		compile_(o);
		it = compiled_ops_.find(o);
		assert(it != compiled_ops_.end());
	    }
	    
	    assert(it->second.out_node);
	    rt_ops.push_back(it->second.out_node);
	}

	//Get list of taks
	std::vector<rt::Node*> rt_tasks = full_rt_graph_.topological_sort(rt_ops);

	//set inut values
	for (auto x : inputs)
	{
	    auto it = compiled_ops_.find(x.first);
	    assert(it != compiled_ops_.end());
	    auto dst = it->second.out_data;
	    input_shapes_[x.first] = x.second.second;
	    tensor_write(dst, dst + x.second.second.total(), x.second.first);
	}
	
	//run computations
	cpu::run_sequential(rt_tasks);

	//set output values
	for (std::size_t i = 0; i < outputs.size(); ++i)
	{
	    dbl_t* out_ptr = outputs[i];
	    auto it = compiled_ops_.find(ops[i]);
	    assert(it != compiled_ops_.end());
	    const dbl_t* src_ptr = it->second.out_data;
	    auto shape = it->second.out_shape;
	    tensor_read(src_ptr, src_ptr + shape.total(), out_ptr);
	}
    }

    void Graph::add_compiled(Op* op, const std::vector<rt::Node*> nodes,
			     std::vector<dbl_t*> tensors,
			     rt::Node* out_node, const Shape& out_shape, dbl_t* out_data)
    {
        assert(compiled_ops_.find(op) == compiled_ops_.end());
	CompiledOp cop(op, nodes, tensors, out_node, out_shape, out_data);
	compiled_ops_.emplace(op, std::move(cop));
	for (auto n : nodes)
	    full_rt_graph_.add(n);
    }

    const CompiledOp& Graph::compiled(Op* op)
    {
	auto it = compiled_ops_.find(op);
	assert(it != compiled_ops_.end());
	return it->second;
    }


    void Graph::remove_compiled_rec_(Op* op)
    {
	auto it = compiled_ops_.find(op);
	if (it == compiled_ops_.end())
	    return;
	
	compiled_ops_.erase(it);
	for (auto x : op->succs())
	    remove_compiled_rec_(x);
    }
	

    void Graph::compile_(Op* node)
    {
	if (compiled_ops_.find(node) != compiled_ops_.end())
	    return;
	
	for (auto pred : node->preds())
	    compile_(pred);

	node->compile();
    }
    
}