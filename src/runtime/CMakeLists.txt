set(SRC
  node.cu
  nodes-list.cu
  graph.cu
  optimizer.cu
)

add_library(runtime_obj OBJECT ${SRC})
