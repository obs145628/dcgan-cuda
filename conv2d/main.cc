#include <iostream>

#include "tensor4.hh"
#include "conv.hh"
#include "conv_simplified.hh"

#include <tocha/tensor.hh>

std::vector<Tensor4> load_tensors(const std::string& path)
{
    auto data = tocha::Tensors::load(path);
    std::vector<Tensor4> res;

    for (auto& tmp: data.arr())
    {
        if (tmp.dims.size() != 4)
            throw std::runtime_error {"not a tensor of size 4"};

        Tensor4 t(tmp.dims[0], tmp.dims[1], tmp.dims[2], tmp.dims[3]);
        auto ptr = reinterpret_cast<float*>(tmp.data);
        std::copy(ptr, ptr + t.size, t.data);
        res.push_back(t);
    }

    return res;
}

void save_tensors(const std::string& path, const std::vector<Tensor4>& tensors)
{
    tocha::Tensors out;

    for (auto& t : tensors)
    {
        out.add(tocha::Tensor::f32(t.d1, t.d2, t.d3, t.d4));
        auto ptr = reinterpret_cast<float*>(out.arr().back().data);
        std::copy(t.data, t.data + t.size, ptr);
    }

    out.save(path);
}

int main()
{
    auto input = load_tensors("./conv_in.npz");
    std::cout << "== input:\n"; 
    for (auto& t : input)
        t.dump_shape();
    std::cout << "==\n\n";

    const Tensor4& x = input[0];
    const Tensor4& k = input[1];
    const Tensor4& dy = input[2];
    (void) k;
    (void) dy;

    std::size_t sh = 2;
    std::size_t sw = 2;
    std::size_t p1 = 1;
    std::size_t p2 = 2;
    std::size_t p3 = 1;
    std::size_t p4 = 2;
    
    Tensor4 y = conv2d_sp(x, k, sh, sw, p1, p2, p3, p4);
    Tensor4 dk = conv2d_sp_dk(x, dy, sh, sw, p1, p2, p3, p4);
    Tensor4 dx = conv2d_sp_dx(k, dy, sh, sw, p1, p2, p3, p4);

    std::vector<Tensor4> output {y, dk, dx};
    std::cout << "== output:\n";
    for (auto& t : output)
        t.dump_shape();
    std::cout << "==\n\n";
    save_tensors("./out.tbin", output);
    

    
}
