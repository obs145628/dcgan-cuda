#include "variable.hh"
#include "graph.hh"
#include "../runtime/graph.hh"
#include "../runtime/node.hh"
#include "../memory/alloc.hh"
#include "../memory/copy.hh"

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

    Variable::Variable(const Shape& shape)
        : Op(shape)
        , data_(tensor_alloc(shape.total()))
        , id_(unique_id())
    {

    }

    Variable::~Variable()
    {
        tensor_free(data_);
    }

    std::size_t Variable::var_id() const
    {
        return id_;
    }

    void Variable::compile()
    {
        Graph::instance().add_compiled(this, {}, {}, nullptr, shape_get(), data_);
    }

    void Variable::write(const dbl_t* ptr)
    {
        tensor_write(data_, data_ + shape_get().total(), ptr);
    }
    
    void Variable::read(dbl_t* ptr) const
    {
        tensor_read(data_, data_ + shape_get().total(), ptr);
    }
    
}
