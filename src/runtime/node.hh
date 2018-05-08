#pragma once

#include <vector>
#include "../memory/types.hh"

namespace rt
{

    struct Node
    {

        static constexpr int OP_MAT_MAT_MUL = 0;
        static constexpr int OP_MAT_RVECT_ADD = 1;
        static constexpr int OP_SIGMOID = 2;
        static constexpr int OP_MSE = 3;
        static constexpr int OP_SOFTMAX = 4;
        static constexpr int OP_LOG_SOFTMAX = 5;
        static constexpr int OP_SOFTMAX_CROSS_ENTROPY = 6;
        static constexpr int OP_CONV2D = 7;
        static constexpr int OP_RELU = 8;
        static constexpr int OP_RELU_LEAKY = 9;
        static constexpr int OP_TANH = 10;
        static constexpr int OP_MSE_GRAD = 11;
        static constexpr int OP_SIGMOID_GRAD = 12;

        static Node* op_conv2d(const dbl_t* input, const dbl_t* kernel, const int strides[],
                               dbl_t* output, const int input_size[], const int kernel_size[],
                               const std::vector<Node*>& preds);

        static Node* op_mat_mat_mul(const dbl_t* left, const dbl_t* right, dbl_t* output,
                                    std::size_t rowsl, std::size_t colsl, std::size_t colsr,
                                    const std::vector<Node*>& preds);

        static Node* op_mat_rvect_add(const dbl_t* left, const dbl_t* right, dbl_t* output,
                                      std::size_t rows, std::size_t cols,
                                      const std::vector<Node*>& preds);

        static Node* op_relu(const dbl_t* args, dbl_t* out, std::size_t len,
                             const std::vector<Node*>& preds);

        static Node* op_relu_leaky(const dbl_t* args, dbl_t* out,
                                   std::size_t len, const dbl_t alpha,
                                   const std::vector<Node*>& preds);

        static Node* op_sigmoid(const dbl_t* args, dbl_t* out, std::size_t len,
                                const std::vector<Node*>& preds);

        static Node* op_mse(const dbl_t* y, const dbl_t* y_hat, dbl_t* out,
                            std::size_t rows, std::size_t cols,
                            const std::vector<Node*>& preds);

        static Node* op_softmax(const dbl_t* args, dbl_t* out,
                                std::size_t rows, std::size_t cols,
                                const std::vector<Node*>& preds);

        static Node* op_log_softmax(const dbl_t* args, dbl_t* out,
                                    std::size_t rows, std::size_t cols,
                                    const std::vector<Node*>& preds);

        static Node* op_softmax_cross_entropy(const dbl_t* y, const dbl_t* logits, dbl_t* out,
                                              std::size_t rows, std::size_t cols,
                                              const std::vector<Node*>& preds);

        static Node* op_mse_grad(const dbl_t* y, const dbl_t* y_hat, dbl_t* out,
                                 std::size_t len,
                                 const std::vector<Node*>& preds);

        static Node* op_tanh(const dbl_t* args, dbl_t* out, std::size_t len,
                             const std::vector<Node*>& preds);

        static Node* op_sigmoid_grad(const dbl_t* sig_out, const dbl_t* dout, dbl_t* out,
                                     std::size_t len,
                                     const std::vector<Node*>& preds);

        Node(int type, std::vector<Node*> preds);
        Node(const Node&) = delete;
        Node& operator=(const Node&) = delete;
        int type;
        std::vector<Node*> preds;
        std::vector<Node*> succs;

        const dbl_t* in1;
        const dbl_t* in2;
        const dbl_t* in3;
        dbl_t* out1;
        dbl_t* out2;
        dbl_t cons1;
        dbl_t cons2;
        dbl_t cons3;
        std::size_t len1;
        std::size_t len2;
        std::size_t len3;
        int intconst[2];
        int sizes1[4];
        int sizes2[4];
        dbl_t alpha_leaky;
    };

}
