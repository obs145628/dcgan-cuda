#include "seq.hh"
#include <cassert>
#include <stdexcept>
#include "graph.hh"
#include "mse-grad.hh"
#include "ops-builder.hh"
#include "../runtime/node.hh"
#include "../memory/alloc.hh"

namespace ops
{

    Seq::Seq(std::vector<Op*> ops)
        : Op("seq", ops.back()->shape_get(), ops)
    {}

    void Seq::compile()
    {
        auto& g = Graph::instance();
        auto& clast = g.compiled(preds().back());

        std::vector<rt::Node*> nodes;
        for (auto p : preds())
            nodes.push_back(g.compiled(p).out_node);

        auto out_node = rt::Node::nop(nodes);
        g.add_compiled(this, {out_node}, {}, out_node, clast.out_shape, clast.out_data);
    }
}
