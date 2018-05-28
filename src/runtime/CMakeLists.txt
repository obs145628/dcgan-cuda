set(SRC
  node.cc
  nodes-list.cc
  graph.cc
  optimizer.cc
)

add_library(runtime_obj OBJECT ${SRC})
