#include "input.hh"
#include "graph.hh"
#include "../runtime/graph.hh"
#include "../runtime/node.hh"
#include "../memory/alloc.hh"

namespace ops
{

    namespace
    {
        std::size_t unique_id()
        {
            static std::size_t res = 0;
            return res++;
        }
    }

    Input::Input(const Shape& shape)
        : Op(shape)
        , id_(unique_id())
        , data_(nullptr)
    {}

    std::size_t Input::input_id() const
    {
        return id_;
    }

    void Input::compile()
    {
        auto& g = Graph::instance();
        auto& inputs = Graph::instance().input_shapes_get();
        auto shape = inputs.find(this)->second;
        data_ = tensor_alloc(shape.total());

        g.add_compiled(this, {}, {data_}, nullptr, shape, data_);
    }
    
}
