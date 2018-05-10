#include "graph.hh"
#include "op.hh"
#include <cassert>
#include "../cpu/runner.hh"
#include "../memory/copy.hh"
#include "input.hh"
#include "variable.hh"

namespace ops
{

    Graph& Graph::instance()
    {
        static Graph graph;
        return graph;
    }

    Graph::Graph()
        : full_rt_graph_()
        , debug_(false)
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

    const std::map<std::string, Op*> Graph::ops_by_name() const
    {
        return ops_by_name_;
    }
    
    const std::vector<Variable*>& Graph::vars_list() const
    {
        return vars_;
    }

    std::vector<Variable*> Graph::train_vars_get(const Op* cost)
    {
        std::vector<Variable*> res;
        for (auto v : vars_)
            if (v->is_trainable() && (!cost || v->pred_of(cost)))
                res.push_back(v);
        return res;
    }

    void Graph::add(Op* op)
    {
        ops_.push_back(op);
        
        if (ops_by_name_.find(op->name_get()) != ops_by_name_.end())
            throw std::runtime_error {"Operand with same name already exists"};
        ops_by_name_[op->name_get()] = op;
    }

    void Graph::add_var(Variable* var)
    {
        add(var);
        vars_.push_back(var);
    }

    const std::map<Input*, Shape>& Graph::input_shapes_get()
    {
        return input_shapes_;
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


        //debug display
        if (debug_)
        {
            rt::Graph::print_nodes(std::cout, rt_tasks);
            auto dot = to_dot_graph();
            dot.write_file("./graph.dot");
            auto rt_dot = full_rt_graph_.to_dot_graph();
            rt_dot.write_file("./rt_graph.dot");
        }

        //run computations
        cpu::run_sequential(rt_tasks);

        //set output values
        for (std::size_t i = 0; i < outputs.size(); ++i)
        {
            dbl_t* out_ptr = outputs[i];
            if (!out_ptr)
                continue;
            
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

    Op* Graph::gradient(Op* out, Op* var)
    {

        auto it = grads_.find({out, var});
        if (it == grads_.end())
        {
            auto res = compute_gradient_(out, var);
            grads_[{out, var}] = res;
            return res;
        }
        else
            return it->second;
    }

    void Graph::debug_set(bool debug)
    {
        debug_ = debug;
    }

    utils::DotGraph Graph::to_dot_graph()
    {
        utils::DotGraph res;
        for (auto op : ops_)
            for (auto succ : op->succs())
                res.add_edge(op->name_get(), succ->name_get());
        return res;
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

    Op* Graph::compute_gradient_(Op* out, Op* var)
    {
        std::size_t vari = out->pred_index(var);
        if (vari != std::size_t(-1))
            return out->child_grad(vari, nullptr);

        Op* succ = var->pred_of(out);
        if (succ == nullptr)
            throw std::runtime_error {"Can't compute the gradient: the nodes are not related"};

        Op* succ_grad = gradient(out, succ);
        vari = succ->pred_index(var);
        assert(vari != std::size_t(-1));
        return succ->child_grad(vari, succ_grad);
    }
    
}
