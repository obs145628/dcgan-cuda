#include "node.hh"

#include <iostream>

namespace rt
{

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
	
	Node* Node::op_conv2d(const dbl_t* input, const dbl_t* mask, dbl_t* output, std::size_t rowsl, std::size_t colsl, const std::vector<Node*>& preds)
    {
	auto res = new Node(OP_CONV2D, preds);
	res->in1 = input;
	res->in2 = mask;
	res->out1 = output;
	res->len1 = rowsl;
	res->len2 = colsl;
	return res;
    }

    Node::Node(int type, std::vector<Node*> preds)
	: type(type)
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
