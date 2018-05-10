#include "node.hh"

#include <iostream>

namespace rt
{

    const char* Node::OP_NAMES[20] =
    {
        "mat_mat_mul",
        "mat_rvect_add",
        "sigmoid",
        "mse",
        "softmax",
        "log_softmax",
        "softmax_cross_entropy",
        "conv2d",
        "relu",
        "relu_leaky",
        "tanh",
        "mse_grad",
        "sigmoid_grad",
        "mat_mul_add",
        "tmat_mat_mul",
        "mat_tmat_mul",
        "mat_sum_rows",
        "mat_sum_cols",
        "softmax_cross_entropy_grad",
        "conv2d_bias_add"
    };
    

    Node* Node::op_mat_mat_mul(const dbl_t* left, const dbl_t* right, dbl_t* output,
                               std::size_t rowsl, std::size_t colsl, std::size_t colsr,
                               const std::vector<Node*>& preds)
    {
        auto res = new Node(OP_MAT_MAT_MUL, preds);
        res->in1 = left;
        res->in2 = right;
        res->out1 = output;
        res->len1 = rowsl;
        res->len2 = colsl;
        res->len3 = colsr;
        return res;
    }

    Node* Node::op_mat_rvect_add(const dbl_t* left, const dbl_t* right, dbl_t* output,
                                 std::size_t rows, std::size_t cols,
                                 const std::vector<Node*>& preds)
    {
        auto res = new Node(OP_MAT_RVECT_ADD, preds);
        res->in1 = left;
        res->in2 = right;
        res->out1 = output;
        res->len1 = rows;
        res->len2 = cols;
        return res;
    }

    Node* Node::op_sigmoid(const dbl_t* args, dbl_t* output, std::size_t len,
                           const std::vector<Node*>& preds)
    {
        auto res = new Node(OP_SIGMOID, preds);
        res->in1 = args;
        res->out1 = output;
        res->len1 = len;
        return res;
    }

    Node* Node::op_mse(const dbl_t* y, const dbl_t* y_hat, dbl_t* out,
                       std::size_t rows, std::size_t cols,
                       const std::vector<Node*>& preds)
    {
        auto res = new Node(OP_MSE, preds);
        res->in1 = y;
        res->in2 = y_hat;
        res->out1 = out;
        res->len1 = rows;
        res->len2 = cols;
        return res;
    }

    Node* Node::op_mse_grad(const dbl_t* y, const dbl_t* y_hat, dbl_t* out,
                            std::size_t len,
                            const std::vector<Node*>& preds)
    {
        auto res = new Node(OP_MSE_GRAD, preds);
        res->in1 = y;
        res->in2 = y_hat;
        res->out1 = out;
        res->len1 = len;
        return res;
    }

    Node* Node::op_conv2d(const dbl_t* input, const dbl_t* kernel,
                          const int strides[], dbl_t* output,
                          const int input_size[], const int kernel_size[],
                          const std::vector<Node*>& preds)
    {
        auto res = new Node(OP_CONV2D, preds);
        res->in1 = input;
        res->in2 = kernel;
        res->intconst[0] = strides[0];
        res->intconst[1] = strides[1];
        res->out1 = output;
        res->sizes1[0] = input_size[0];
        res->sizes1[1] = input_size[1];
        res->sizes1[2] = input_size[2];
        res->sizes1[3] = input_size[3];
        res->sizes2[0] = kernel_size[0];
        res->sizes2[1] = kernel_size[1];
        res->sizes2[2] = kernel_size[2];
        res->sizes2[3] = kernel_size[3];
        return res;
    }
    
    Node* Node::op_conv2d_bias_add(const dbl_t* z, const dbl_t* bias, dbl_t* output,
                             const int input_size[], const std::vector<Node*>& preds)
    {
        auto res = new Node(OP_CONV2D_BIAS_ADD, preds);
        res->in1 = z;
        res->in2 = bias;
        res->out1 = output;
        res->sizes1[0] = input_size[0];
        res->sizes1[1] = input_size[1];
        res->sizes1[2] = input_size[2];
        res->sizes1[3] = input_size[3];
        return res;
    }

    Node* Node::op_softmax(const dbl_t* args, dbl_t* output,
                           std::size_t rows, std::size_t cols,
                           const std::vector<Node*>& preds)
    {
        auto res = new Node(OP_SOFTMAX, preds);
        res->in1 = args;
        res->out1 = output;
        res->len1 = rows;
        res->len2 = cols;
        return res;
    }

    Node* Node::op_log_softmax(const dbl_t* args, dbl_t* out,
                               std::size_t rows, std::size_t cols,
                               const std::vector<Node*>& preds)
    {
        auto res = new Node(OP_LOG_SOFTMAX, preds);
        res->in1 = args;
        res->out1 = out;
        res->len1 = rows;
        res->len2 = cols;
        return res;
    }

    Node* Node::op_softmax_cross_entropy(const dbl_t* y, const dbl_t* logits, dbl_t* out,
                                         std::size_t rows, std::size_t cols,
                                         const std::vector<Node*>& preds)
    {
        auto res = new Node(OP_SOFTMAX_CROSS_ENTROPY, preds);
        res->in1 = y;
        res->in2 = logits;
        res->out1 = out;
        res->len1 = rows;
        res->len2 = cols;
        return res;
    }

    Node* Node::op_relu(const dbl_t* args, dbl_t* output, std::size_t len,
                        const std::vector<Node*>& preds)
    {
        auto res = new Node(OP_RELU, preds);
        res->in1 = args;
        res->out1 = output;
        res->len1 = len;
        return res;
    }

    Node* Node::op_relu_leaky(const dbl_t* args, dbl_t* output, std::size_t len,
                              const dbl_t alpha,
                              const std::vector<Node*>& preds)
    {
        auto res = new Node(OP_RELU_LEAKY, preds);
        res->in1 = args;
        res->out1 = output;
        res->len1 = len;
        res->alpha_leaky = alpha;
        return res;
    }

    Node* Node::op_tanh(const dbl_t* args, dbl_t* output, std::size_t len,
                        const std::vector<Node*>& preds)
    {
        auto res = new Node(OP_TANH, preds);
        res->in1 = args;
        res->out1 = output;
        res->len1 = len;
        return res;
    }

    Node* Node::op_sigmoid_grad(const dbl_t* sig_out, const dbl_t* dout, dbl_t* out,
                                std::size_t len,
                                const std::vector<Node*>& preds)
    {
        auto res = new Node(OP_SIGMOID_GRAD, preds);
        res->in1 = sig_out;
        res->in2 = dout;
        res->out1 = out;
        res->len1 = len;
        return res;
    }

    Node* Node::op_mat_mul_add(const dbl_t* x, const dbl_t* w, const dbl_t* b,
                               dbl_t* output,
                               std::size_t rowsx, std::size_t colsx, std::size_t colsw,
                               const std::vector<Node*>& preds)
    {
        auto res = new Node(OP_MAT_MUL_ADD, preds);
        res->in1 = x;
        res->in2 = w;
        res->in3 = b;
        res->out1 = output;
        res->len1 = rowsx;
        res->len2 = colsx;
        res->len3 = colsw;
        return res;
    }

    Node* Node::op_tmat_mat_mul(const dbl_t* left, const dbl_t* right, dbl_t* output,
                                std::size_t rowsl, std::size_t colsl, std::size_t colsr,
                                const std::vector<Node*>& preds)
    {
        auto res = new Node(OP_TMAT_MAT_MUL, preds);
        res->in1 = left;
        res->in2 = right;
        res->out1 = output;
        res->len1 = rowsl;
        res->len2 = colsl;
        res->len3 = colsr;
        return res;
    }

    Node* Node::op_mat_tmat_mul(const dbl_t* left, const dbl_t* right, dbl_t* output,
                                std::size_t rowsl, std::size_t colsl, std::size_t colsr,
                                const std::vector<Node*>& preds)
    {
        auto res = new Node(OP_MAT_TMAT_MUL, preds);
        res->in1 = left;
        res->in2 = right;
        res->out1 = output;
        res->len1 = rowsl;
        res->len2 = colsl;
        res->len3 = colsr;
        return res;
    }

    Node* Node::op_mat_sum_rows(const dbl_t* arg, dbl_t* out,
                                std::size_t rows, std::size_t cols,
                                const std::vector<Node*>& preds)
    {
        auto res = new Node(OP_MAT_SUM_ROWS, preds);
        res->in1 = arg;
        res->out1 = out;
        res->len1 = rows;
        res->len2 = cols;
        return res;
    }

    Node* Node::op_mat_sum_cols(const dbl_t* arg, dbl_t* out,
                                std::size_t rows, std::size_t cols,
                                const std::vector<Node*>& preds)
    {
        auto res = new Node(OP_MAT_SUM_COLS, preds);
        res->in1 = arg;
        res->out1 = out;
        res->len1 = rows;
        res->len2 = cols;
        return res;
    }

    Node* Node::op_softmax_cross_entropy_grad(const dbl_t* y, const dbl_t* logits, dbl_t* out,
                                              std::size_t rows, std::size_t cols,
                                              const std::vector<Node*>& preds)
    {
        auto res = new Node(OP_SOFTMAX_CROSS_ENTROPY_GRAD, preds);
        res->in1 = y;
        res->in2 = logits;
        res->out1 = out;
        res->len1 = rows;
        res->len2 = cols;
        return res;
    }

    Node::Node(int type, std::vector<Node*> preds)
        : type(type)
        , in1(nullptr)
        , in2(nullptr)
        , in3(nullptr)
        , out1(nullptr)
        , out2(nullptr)
        , len1(0)
        , len2(0)
        , len3(0)
    {
        for (auto n : preds)
        {
            if (n)
            {
                n->succs.push_back(this);
                this->preds.push_back(n);
            }
        }
    }

}
