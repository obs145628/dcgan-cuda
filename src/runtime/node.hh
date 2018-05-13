#pragma once

#include <vector>
#include "../memory/types.hh"

namespace rt
{

    struct Node
    {

        /*
         * used only because it's predecessor must be executed
         * never kept in exec list
         */
        static constexpr int OP_NOP = 1000;

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
        static constexpr int OP_MAT_MUL_ADD = 13;
        static constexpr int OP_TMAT_MAT_MUL = 14;
        static constexpr int OP_MAT_TMAT_MUL = 15;
        static constexpr int OP_MAT_SUM_ROWS = 16;
        static constexpr int OP_MAT_SUM_COLS = 17;
        static constexpr int OP_SOFTMAX_CROSS_ENTROPY_GRAD = 18;
        static constexpr int OP_RELU_GRAD = 19;
        static constexpr int OP_CONV2D_BIAS_ADD = 20;
        static constexpr int OP_UPDATE = 21;
        static constexpr int OP_SIGMOID_CROSS_ENTROPY = 22;
        static constexpr int OP_SIGMOID_CROSS_ENTROPY_GRAD = 23;
        static constexpr int OP_CONV2D_INPUT_GRAD = 24;
        static constexpr int OP_CONV2D_KERNEL_GRAD = 25;
        static constexpr int OP_ARGMAX_ACC = 26;
        static constexpr int OP_MOMENT_UPDATE = 27;
        static constexpr int OP_MOMENT_UPDATE2 = 28;
        static constexpr int OP_ADAM_UPDATE = 29;
        static constexpr int OP_LEAKY_RELU_GRAD = 30;
        static constexpr int OP_CONV2D_BIAS_ADD_GRAD = 31;
        static constexpr int OP_TANH_GRAD = 32;
        static constexpr int OP_CONV2D_TRANSPOSE = 33;

        static const char* OP_NAMES[34];

        static Node* nop(const std::vector<Node*>& preds);

        static Node* op_conv2d(const dbl_t* input, const dbl_t* kernel, const int strides[],
                               int pad_top, int pad_left, dbl_t* output,
                               const int input_size[], const int kernel_size[],
                               const std::vector<Node*>& preds);

        static Node* op_conv2d_bias_add(const dbl_t* z, const dbl_t* bias, dbl_t* output,
                                        const int input_size[], const std::vector<Node*>& preds);

        static Node* op_conv2d_bias_add_grad(const dbl_t* z, const int size[],
                                             dbl_t* output,
                                             const std::vector<Node*>& preds);

        static Node* op_conv2d_input_grad(const dbl_t* y, const dbl_t* kernel, const int strides[],
                                          dbl_t* output, const int y_size[], const int kernel_size[],
                                          const int input_size[],
                                          const std::vector<Node*>& preds);

        static Node* op_conv2d_kernel_grad(const dbl_t* y, const dbl_t* input, const int strides[],
                                          dbl_t* output, const int y_size[], const int input_size[],
                                          const int padded_size[],
                                          const std::vector<Node*>& preds);

        static Node* op_conv2d_transpose(const dbl_t* input, const dbl_t* kernel, const int out_size[],
                                         const int strides[], dbl_t* output,
                                         const int input_size[], const int kernel_size[],
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

        static Node* op_mat_mul_add(const dbl_t* x, const dbl_t* w, const dbl_t* b,
                                    dbl_t* output,
                                    std::size_t rowsx, std::size_t colsx, std::size_t colsw,
                                    const std::vector<Node*>& preds);

        static Node* op_tmat_mat_mul(const dbl_t* left, const dbl_t* right, dbl_t* output,
                                     std::size_t rowsl, std::size_t colsl, std::size_t colsr,
                                     const std::vector<Node*>& preds);

        static Node* op_mat_tmat_mul(const dbl_t* left, const dbl_t* right, dbl_t* output,
                                     std::size_t rowsl, std::size_t colsl, std::size_t colsr,
                                     const std::vector<Node*>& preds);

        static Node* op_mat_sum_rows(const dbl_t* arg, dbl_t* out,
                                     std::size_t rows, std::size_t cols,
                                     const std::vector<Node*>& preds);

        static Node* op_mat_sum_cols(const dbl_t* arg, dbl_t* out,
                                     std::size_t rows, std::size_t cols,
                                     const std::vector<Node*>& preds);

        static Node* op_softmax_cross_entropy_grad(const dbl_t* y, const dbl_t* logits, dbl_t* out,
                                                   std::size_t rows, std::size_t cols,
                                                   const std::vector<Node*>& preds);

        static Node* op_relu_grad(const dbl_t* z, const dbl_t* dout, dbl_t* out,
                                  std::size_t len,
                                  const std::vector<Node*>& preds);

        static Node* op_update(dbl_t* var, const dbl_t* dt, const dbl_t* coeff,
                               std::size_t len,
                               const std::vector<Node*>& preds);

        static Node* op_sigmoid_cross_entropy(const dbl_t* y, const dbl_t* logits, dbl_t* out,
                                              std::size_t len,
                                              const std::vector<Node*>& preds);

        static Node* op_sigmoid_cross_entropy_grad(const dbl_t* y, const dbl_t* logits, dbl_t* out,
                                                   std::size_t len,
                                                   const std::vector<Node*>& preds);

        static Node* op_tanh_grad(const dbl_t* tanh_out, const dbl_t* dout, dbl_t* out,
                                     std::size_t len,
                                     const std::vector<Node*>& preds);

        static Node* op_argmax_acc(const dbl_t* y, const dbl_t* y_hat, dbl_t* out,
                                   std::size_t rows, std::size_t cols,
                                   const std::vector<Node*>& preds);

        static Node* op_moment_update(dbl_t* var, const dbl_t* dt,
                                      dbl_t coeff1, dbl_t coeff2, std::size_t len,
                                      const std::vector<Node*>& preds);

        static Node* op_moment_update2(dbl_t* var, const dbl_t* dt,
                                       dbl_t coeff1, dbl_t coeff2, std::size_t len,
                                       const std::vector<Node*>& preds);

        static Node* op_adam_update(dbl_t* var, dbl_t* t, const dbl_t* m, const dbl_t* v,
                                    dbl_t lr, dbl_t beta1, dbl_t beta2, dbl_t eps,
                                    std::size_t len,
                                    const std::vector<Node*>& preds);

         static Node* op_leaky_relu_grad(const dbl_t* z, const dbl_t* dout, dbl_t* out,
                                         dbl_t alpha, std::size_t len,
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
        dbl_t cons4;
        std::size_t len1;
        std::size_t len2;
        std::size_t len3;
        int intconst[2];
        int intconst2[2];
        int sizes1[4];
        int sizes2[4];
        int sizes3[4];
        int int_cons1;
        int int_cons2;
        dbl_t alpha_leaky;
    };

}
