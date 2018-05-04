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

	static Node* op_mat_mat_mul(const dbl_t* left, const dbl_t* right, dbl_t* output,
				    std::size_t rowsl, std::size_t colsl, std::size_t colsr,
				    const std::vector<Node*>& preds);

	static Node* op_mat_rvect_add(const dbl_t* left, const dbl_t* right, dbl_t* output,
				      std::size_t rows, std::size_t cols,
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
    };

}
