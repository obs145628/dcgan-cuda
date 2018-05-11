#include <stdexcept>

#include "reshape.hh"
#include "graph.hh"
#include "../runtime/graph.hh"
#include "../runtime/node.hh"
#include "../memory/alloc.hh"
#include "ops-builder.hh"
#include <cassert>
#include <stdexcept>

namespace ops
{

    Reshape::Reshape(Op* arg, const Shape& shape)
        : Op("reshape", shape, {arg})
        , m_initial_size(arg->shape_get())
    {}

    void Reshape::compile()
    {
        auto& g = Graph::instance();
        auto& carg = g.compiled(preds()[0]);
        auto& new_shape = shape_get();

        if (new_shape.defined())
            g.add_compiled(this, {}, {}, nullptr, new_shape, carg.out_data);
        else
        {
            auto& carg_shape = carg.out_shape;
            std::vector<int> new_dims;
            for (auto x : new_shape.dims())
                if (x == -1)
                    new_dims.push_back((int) (carg_shape.total() / (- new_shape.total())));
                else
                    new_dims.push_back(x);
            g.add_compiled(this, {}, {}, nullptr, Shape(new_dims), carg.out_data);
        }
    }

    Op* Reshape::child_grad(std::size_t index, Op* dout)
    {
        assert(index < 1);
        if (dout != nullptr)
            throw std::runtime_error {"reshape must be the final node of the gradient"};

        auto& builder = OpsBuilder::instance();

        return builder.reshape(preds()[0], m_initial_size);
    }
}
