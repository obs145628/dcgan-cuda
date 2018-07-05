import sys
import numpy as np

output = open(sys.argv[1], "w")

def gen_mat(name, m, n=None):
    output.write("dbl_t " + name + " [] = {\n")

    np.random.seed(3531354)
    if n:
        mat = np.random.rand(m, n)
        for row in mat:
            output.write("\t" + ", ".join([str(x) for x in row]) + ",\n")
    else:
        vec =  np.random.rand(m)
        output.write("\t" + ", ".join([str(x) for x in vec]) + ",\n")

    output.write("};\n\n")

gen_mat("a", 8000, 3)
gen_mat("b", 8000, 2)
gen_mat("v", 8000)