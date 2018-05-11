#include <iostream>
#include <cmath>

#include "../src/utils/dot-graph.hh"

int main()
{

    utils::DotGraph g;
    g.add_edge("t1", "t2");
    g.add_edge("t1", "t3");
    g.add_edge("t2", "t3");
    g.write_file("out.dot");
}
