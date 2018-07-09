set(SRC
  arguments.cu
  date.cu
  dot-graph.cu
  dot-tree.cu
  xorshift.cu
)

add_library(utils_obj OBJECT ${SRC})
