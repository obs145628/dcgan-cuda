#include "optimizer.hh"
#include <cassert>
#include "graph.hh"
#include "node.hh"

namespace rt
{

    namespace
    {

        //can't divide if the divide size is <= MIN_NODES_SIZE
        constexpr std::size_t MIN_NODES_SIZE = 1024;

        //max number of parallel nodes when dividing operation
        constexpr std::size_t MAX_NODES = 8;

        //make sure every size of new ops are multiple of this
        constexpr std::size_t SIZE_DIVISOR = 8;

        void elemwhise_size(std::size_t total, std::size_t& divs, std::size_t& size)
        {
            std::size_t n = total / MIN_NODES_SIZE;
            n = std::min(n, MAX_NODES);
            n = std::max(1UL, n);
            std::size_t m = total / n;
            while (n * m < total)
                ++m;

            if (n > 1)
            {
                while (m % SIZE_DIVISOR != 0)
                    ++m;
            }

            divs = n;
            size = m;
        }
        

        using opti_f = Node* (*)(Graph&, Node*, const std::vector<Node*>&);

        Node* opti_mat_mat_mul(Graph& graph, Node* node, const std::vector<Node*>& preds)
        {
            const std::size_t nm = node->len1;
            const std::size_t nn = node->len2;
            const std::size_t np = node->len3;

            std::size_t n;
            std::size_t m;

            if (nm > np) {
                elemwhise_size(nm, n, m);
            } else {
                elemwhise_size(np, n, m);
            }

            if (n < 2)
            {
                auto res = Node::op_mat_mat_mul(node->in1, node->in2, node->out1,
                    node->len1, node->len2, node->len3, preds);
                graph.add(res);
                return res;
            }

            std::vector<Node*> div_nodes;
            if (node->len1 > node->len2) {
                for (std::size_t i = 0; i < n - 1; ++i)
                    div_nodes.push_back(Node::op_mat_mat_mul(node->in1 + i * m * nn, node->in2,
                                node->out1 + i * m * np, m, nn, np, preds));

                    div_nodes.push_back(Node::op_mat_mat_mul(node->in1 + (n - 1) * m * nn, node->in2,
                                node->out1 + (n - 1) * m * np, nm - (n - 1) * m, nn, np, preds));
            } else {
                for (std::size_t i = 0; i < n - 1; ++i)
                    div_nodes.push_back(Node::op_mat_mat_mul(node->in1, node->in2 + i * m,
                                node->out1 + i * m * np, nm, nn, m, preds));

                    div_nodes.push_back(Node::op_mat_mat_mul(node->in1, node->in2 + (n - 1) * m,
                                node->out1 + (n - 1) * m * np, nm, nn, np - (n - 1) * m, preds));
            }

            for (auto n : div_nodes)
                    graph.add(n);
            
            auto res = Node::nop(div_nodes);
            graph.add(res);
            return res;
        }

        Node* opti_mat_rvect_add(Graph& graph, Node* node, const std::vector<Node*>& preds)
        {
            std::size_t n;
            std::size_t m;
            elemwhise_size(node->len1, n, m);

            if (n < 2)
            {
                auto res = Node::op_mat_rvect_add(node->in1, node->in2, node->out1,
                    node->len1, node->len2, preds);
                graph.add(res);
                return res;
            }

            const std::size_t nv = node->len2;

            std::vector<Node*> div_nodes;
            for (std::size_t i = 0; i < n - 1; ++i)
                div_nodes.push_back(Node::op_mat_rvect_add(node->in1 + (i * m * nv), node->in2,
                                                node->out1 + (i * m * nv), m, nv, preds));

            div_nodes.push_back(Node::op_mat_rvect_add(node->in1 + (n - 1) * m * nv, node->in2,
                node->out1 + (n - 1) * m * nv, node->len1 - ((n - 1) * m), nv, preds));

            for (auto n : div_nodes)
                graph.add(n);

            auto res = Node::nop(div_nodes);
            graph.add(res);
            return res;
        }

        Node* opti_sigmoid(Graph& graph, Node* node, const std::vector<Node*>& preds)
        {
            std::size_t n;
            std::size_t m;
            elemwhise_size(node->len1, n, m);

            if (n < 2)
            {
                auto res = Node::op_sigmoid(node->in1, node->out1,
                                            node->len1, preds);
                //res->use_simd = node->len1 % SIZE_DIVISOR == 0;
                    
                graph.add(res);
                return res;
            }

            const dbl_t* in = node->in1;
            dbl_t* out = node->out1;

            std::vector<Node*> div_nodes;
            for (std::size_t i = 0; i < n - 1; ++i)
                div_nodes.push_back(Node::op_sigmoid(in + i * m, out + i * m,
                                                     m, preds));

            div_nodes.push_back(Node::op_sigmoid(in + (n - 1) * m, out + (n - 1) * m,
                                                node->len1 - ((n - 1) * m), preds));

            for (auto n : div_nodes)
            {
                //n->use_simd = true;
                graph.add(n);
            }

            auto res = Node::nop(div_nodes);
            graph.add(res);
            return res;
        }

        Node* opti_mse(Graph& graph, Node* node, const std::vector<Node*>& preds)
        {
            auto res = Node::op_mse(node->in1, node->in2, node->out1,
                                    node->len1, node->len2, preds);
            graph.add(res);
            return res;
        }

        Node* opti_softmax(Graph& graph, Node* node, const std::vector<Node*>& preds)
        {
            std::size_t n;
            std::size_t m;
            elemwhise_size(node->len1, n, m);

            if (n < 2)
            {
                auto res = Node::op_softmax(node->in1, node->out1, node->len1, node->len2, preds);
                graph.add(res);
                return res;
            }

            std::size_t nv = node->len2;

            std::vector<Node*> div_nodes;
            for (std::size_t i = 0; i < n - 1; ++i)
                div_nodes.push_back(Node::op_softmax(node->in1 + (i * m * nv), node->out1 + (i * m * nv),
                                                                        m, nv, preds));

            div_nodes.push_back(Node::op_softmax(node->in1 + (n - 1) * m * nv,
                        node->out1 + (n - 1) * m * nv, node->len1 - ((n - 1) * m), nv, preds));

            for (auto n : div_nodes)
                graph.add(n);

            auto res = Node::nop(div_nodes);
            graph.add(res);
            return res;
        }

        Node* opti_log_softmax(Graph& graph, Node* node, const std::vector<Node*>& preds)
        {
            std::size_t n;
            std::size_t m;
            elemwhise_size(node->len1, n, m);

            if (n < 2)
            {
                auto res = Node::op_log_softmax(node->in1, node->out1, node->len1, node->len2, preds);
                graph.add(res);
                return res;
            }

            std::size_t nv = node->len2;

            std::vector<Node*> div_nodes;
            for (std::size_t i = 0; i < n - 1; ++i)
                div_nodes.push_back(Node::op_log_softmax(node->in1 + (i * m * nv),
                                            node->out1 + (i * m * nv), m, nv, preds));

            div_nodes.push_back(Node::op_log_softmax(node->in1 + (n - 1) * m * nv,
                                        node->out1 + (n - 1) * m * nv, node->len1 - ((n - 1) * m), nv, preds));

            for (auto n : div_nodes)
                graph.add(n);

            auto res = Node::nop(div_nodes);
            graph.add(res);
            return res;
        }

        Node* opti_softmax_cross_entropy(Graph& graph, Node* node, const std::vector<Node*>& preds)
        {
            auto res = Node::op_softmax_cross_entropy(node->in1, node->in2, node->out1,
                                                      node->len1, node->len2, preds);
            graph.add(res);
            return res;
        }

        Node* opti_conv2d(Graph& graph, Node* node, const std::vector<Node*>& preds)
        {
            auto res = Node::op_conv2d(node->in1, node->in2, node->intconst,
                                       node->int_cons1, node->int_cons2, node->out1,
                                       node->sizes1, node->sizes2, preds);
            graph.add(res);
            return res;
        }

        Node* opti_relu(Graph& graph, Node* node, const std::vector<Node*>& preds)
        {
            std::size_t n;
            std::size_t m;
            elemwhise_size(node->len1, n, m);

            if (n < 2)
            {
                auto res = Node::op_relu(node->in1, node->out1,
                                            node->len1, preds);
                graph.add(res);
                return res;
            }

            const dbl_t* in = node->in1;
            dbl_t* out = node->out1;

            std::vector<Node*> div_nodes;
            for (std::size_t i = 0; i < n - 1; ++i)
                div_nodes.push_back(Node::op_relu(in + i * m, out + i * m, m, preds));

            div_nodes.push_back(Node::op_relu(in + (n - 1) * m, out + (n - 1) * m,
                                                node->len1 - (n - 1) * m, preds));

            for (auto n : div_nodes)
                graph.add(n);

            auto res = Node::nop(div_nodes);
            graph.add(res);
            return res;
        }

        Node* opti_relu_leaky(Graph& graph, Node* node, const std::vector<Node*>& preds)
        {
            std::size_t n;
            std::size_t m;
            elemwhise_size(node->len1, n, m);

            if (n < 2)
            {
                auto res = Node::op_relu_leaky(node->in1, node->out1,
                                            node->len1, node->alpha_leaky, preds);
                graph.add(res);
                return res;
            }

            const dbl_t* in = node->in1;
            dbl_t* out = node->out1;

            std::vector<Node*> div_nodes;
            for (std::size_t i = 0; i < n - 1; ++i)
                div_nodes.push_back(Node::op_relu_leaky(in + i * m, out + i * m,
                                                     m, node->alpha_leaky, preds));

            div_nodes.push_back(Node::op_relu_leaky(in + (n - 1) * m, out + (n - 1) * m,
                                                 node->len1 - (n - 1) * m, node->alpha_leaky, preds));

            for (auto n : div_nodes)
                graph.add(n);

            auto res = Node::nop(div_nodes);
            graph.add(res);
            return res;
        }

        Node* opti_tanh(Graph& graph, Node* node, const std::vector<Node*>& preds)
        {
            std::size_t n;
            std::size_t m;
            elemwhise_size(node->len1, n, m);

            if (n < 2)
            {
                auto res = Node::op_tanh(node->in1, node->out1,
                                            node->len1, preds);
                graph.add(res);
                return res;
            }

            const dbl_t* in = node->in1;
            dbl_t* out = node->out1;

            std::vector<Node*> div_nodes;
            for (std::size_t i = 0; i < n - 1; ++i)
                div_nodes.push_back(Node::op_tanh(in + i * m, out + i * m,
                                                     m, preds));

            div_nodes.push_back(Node::op_tanh(in + (n - 1) * m, out + (n - 1) * m,
                                                    node->len1 - (n - 1) * m, preds));

            for (auto n : div_nodes)
                graph.add(n);

            auto res = Node::nop(div_nodes);
            graph.add(res);
            return res;
        }

        Node* opti_mse_grad(Graph& graph, Node* node, const std::vector<Node*>& preds)
        {
            std::size_t n;
            std::size_t m;
            elemwhise_size(node->len1, n, m);

            if (n < 2)
            {
                auto res = Node::op_mse_grad(node->in1, node->in2, node->out1,
                                                        node->len1, preds);
                graph.add(res);
                return res;
            }

            std::vector<Node*> div_nodes;
            for (std::size_t i = 0; i < n - 1; ++i)
                div_nodes.push_back(Node::op_mse_grad(node->in1 + i * m, node->in2 + i * m,
                                                            node->out1 + i * m, m, preds));

            div_nodes.push_back(Node::op_mse_grad(node->in1 + (n - 1) * m, node->in2 + (n - 1) * m,
                                            node->out1 + (n - 1) * m, node->len1 - (n - 1) * m, preds));

            for (auto n : div_nodes)
                graph.add(n);

            auto res = Node::nop(div_nodes);
            graph.add(res);
            return res;
        }

        Node* opti_sigmoid_grad(Graph& graph, Node* node, const std::vector<Node*>& preds)
        {
            std::size_t n;
            std::size_t m;
            elemwhise_size(node->len1, n, m);

            if (n < 2)
            {
                auto res = Node::op_sigmoid_grad(node->in1, node->in2, node->out1,
                                                        node->len1, preds);
                graph.add(res);
                return res;
            }

            std::vector<Node*> div_nodes;
            for (std::size_t i = 0; i < n - 1; ++i)
                div_nodes.push_back(Node::op_sigmoid_grad(node->in1 + i * m, node->in2 + i * m,
                                                            node->out1 + i * m, m, preds));

            div_nodes.push_back(Node::op_sigmoid_grad(node->in1 + (n - 1) * m, node->in2 + (n - 1) * m,
                                                node->out1 + (n - 1) * m, node->len1 - (n - 1) * m, preds));

            for (auto n : div_nodes)
                graph.add(n);

            auto res = Node::nop(div_nodes);
            graph.add(res);
            return res;
        }

        Node* opti_mat_mul_add(Graph& graph, Node* node, const std::vector<Node*>& preds)
        {
            auto res = Node::op_mat_mul_add(node->in1, node->in2, node->in3,
                                            node->out1, node->len1, node->len2, node->len3,
                                            preds);
            graph.add(res);
            return res;
        }

        Node* opti_tmat_mat_mul(Graph& graph, Node* node, const std::vector<Node*>& preds)
        {
            auto res = Node::op_tmat_mat_mul(node->in1, node->in2, node->out1,
                                             node->len1, node->len2, node->len3, preds);
            graph.add(res);
            return res;
        }

        Node* opti_mat_tmat_mul(Graph& graph, Node* node, const std::vector<Node*>& preds)
        {
            auto res = Node::op_mat_tmat_mul(node->in1, node->in2, node->out1,
                                             node->len1, node->len2, node->len3, preds);
            graph.add(res);
            return res;
        }

        Node* opti_mat_sum_rows(Graph& graph, Node* node, const std::vector<Node*>& preds)
        {
            auto res = Node::op_mat_sum_rows(node->in1, node->out1,
                                             node->len1, node->len2, preds);
            graph.add(res);
            return res;
        }

        Node* opti_mat_sum_cols(Graph& graph, Node* node, const std::vector<Node*>& preds)
        {
            auto res = Node::op_mat_sum_cols(node->in1, node->out1,
                                             node->len1, node->len2, preds);
            graph.add(res);
            return res;
        }

        Node* opti_softmax_cross_entropy_grad(Graph& graph, Node* node, const std::vector<Node*>& preds)
        {
            auto res = Node::op_softmax_cross_entropy_grad(node->in1, node->in2, node->out1,
                                                           node->len1, node->len2, preds);
            graph.add(res);
            return res;
        }

        Node* opti_relu_grad(Graph& graph, Node* node, const std::vector<Node*>& preds)
        {
            std::size_t n;
            std::size_t m;
            elemwhise_size(node->len1, n, m);

            if (n < 2)
            {
                auto res = Node::op_relu_grad(node->in1, node->in2, node->out1,
                                                        node->len1, preds);
                graph.add(res);
                return res;
            }

            std::vector<Node*> div_nodes;
            for (std::size_t i = 0; i < n - 1; ++i)
                div_nodes.push_back(Node::op_relu_grad(node->in1 + i * m, node->in2 + i * m,
                                                            node->out1 + i * m, m, preds));

            div_nodes.push_back(Node::op_relu_grad(node->in1 + (n - 1) * m, node->in2 + (n - 1) * m,
                                            node->out1 + (n - 1) * m, node->len1 - (n - 1) * m, preds));

            for (auto n : div_nodes)
                graph.add(n);

            auto res = Node::nop(div_nodes);
            graph.add(res);
            return res;
        }

        Node* opti_conv2d_bias_add(Graph& graph, Node* node, const std::vector<Node*>& preds)
        {
            auto res = Node::op_conv2d_bias_add(node->in1, node->in2, node->out1,
                                                node->sizes1, preds);
            graph.add(res);
            return res;
        }

        Node* opti_update(Graph& graph, Node* node, const std::vector<Node*>& preds)
        {
            std::size_t n;
            std::size_t m;
            elemwhise_size(node->len1, n, m);

            if (n < 2)
            {
                auto res = Node::op_update(node->out1, node->in1, node->in2,
                                                            node->len1, preds);
                graph.add(res);
                return res;
            }

            std::vector<Node*> div_nodes;
            for (std::size_t i = 0; i < n - 1; ++i)
                div_nodes.push_back(Node::op_update(node->out1 + i * m, node->in1 + i * m, node->in2, m, preds));

            div_nodes.push_back(Node::op_update(node->out1 + (n - 1) * m, node->in1 + (n - 1) * m,
                                                            node->in2, node->len1 - (n - 1) * m, preds));

            for (auto n : div_nodes)
                graph.add(n);

            auto res = Node::nop(div_nodes);
            graph.add(res);
            return res;
        }

        Node* opti_sigmoid_cross_entropy(Graph& graph, Node* node, const std::vector<Node*>& preds)
        {
            auto res = Node::op_sigmoid_cross_entropy(node->in1, node->in2, node->out1,
                                                      node->len1, preds);
            graph.add(res);
            return res;
        }

        Node* opti_sigmoid_cross_entropy_grad(Graph& graph, Node* node,
                                              const std::vector<Node*>& preds)
        {
            std::size_t n;
            std::size_t m;
            elemwhise_size(node->len1, n, m);

            if (n < 2)
            {
                auto res = Node::op_sigmoid_cross_entropy_grad(node->in1, node->in2, node->out1,
                                                        node->len1, preds);
                graph.add(res);
                return res;
            }

            std::vector<Node*> div_nodes;
            for (std::size_t i = 0; i < n - 1; ++i)
                div_nodes.push_back(Node::op_sigmoid_cross_entropy_grad(node->in1 + i * m, node->in2 + i * m,
                                                            node->out1 + i * m, m, preds));

            div_nodes.push_back(Node::op_sigmoid_cross_entropy_grad(node->in1 + (n - 1) * m, node->in2 + (n - 1) * m,
                                                                    node->out1 + (n - 1) * m, node->len1 - (n - 1) * m, preds));

            for (auto n : div_nodes)
                graph.add(n);

            auto res = Node::nop(div_nodes);
            graph.add(res);
            return res;
        }

        Node* opti_conv2d_input_grad(Graph& graph, Node* node, const std::vector<Node*>& preds)
        {
            int input_size[] = {0, node->intconst2[0], node->intconst2[1]};
            
            auto res = Node::op_conv2d_input_grad(node->in1, node->in2,
                                                  node->intconst, node->out1,
                                                  node->sizes1, node->sizes2,
                                                  input_size, preds);
            graph.add(res);
            return res;
        }

        Node* opti_conv2d_kernel_grad(Graph& graph, Node* node, const std::vector<Node*>& preds)
        {
            auto res = Node::op_conv2d_kernel_grad(node->in1, node->in2,
                                                   node->intconst, node->out1,
                                                   node->sizes1, node->sizes2,
                                                   node->intconst2, preds);
            graph.add(res);
            return res;
        }

        Node* opti_argmax_acc(Graph& graph, Node* node, const std::vector<Node*>& preds)
        {
            auto res = Node::op_argmax_acc(node->in1, node->in2, node->out1,
                                           node->len1, node->len2, preds);
            graph.add(res);
            return res;
        }

        Node* opti_moment_update(Graph& graph, Node* node, const std::vector<Node*>& preds)
        {
            std::size_t n;
            std::size_t m;
            elemwhise_size(node->len1, n, m);

            if (n < 2)
            {
                auto res = Node::op_moment_update(node->out1, node->in1, node->cons1,
                                                        node->cons2, node->len1, preds);
                graph.add(res);
                return res;
            }

            std::vector<Node*> div_nodes;
            for (std::size_t i = 0; i < n - 1; ++i)
                div_nodes.push_back(Node::op_moment_update(node->out1 + i * m, node->in1 + i * m,
                                                                node->cons1, node->cons2, m, preds));

            div_nodes.push_back(Node::op_moment_update(node->out1  + (n - 1) * m, node->in1  + (n - 1) * m,
                                                                node->cons1, node->cons2, node->len1 - (n - 1) * m, preds));

            for (auto n : div_nodes)
                graph.add(n);

            auto res = Node::nop(div_nodes);
            graph.add(res);
            return res;
        }

        Node* opti_moment_update2(Graph& graph, Node* node, const std::vector<Node*>& preds)
        {
            std::size_t n;
            std::size_t m;
            elemwhise_size(node->len1, n, m);

            if (n < 2)
            {
                auto res = Node::op_moment_update2(node->out1, node->in1, node->cons1,
                                                        node->cons2, node->len1, preds);
                graph.add(res);
                return res;
            }

            std::vector<Node*> div_nodes;
            for (std::size_t i = 0; i < n - 1; ++i)
                div_nodes.push_back(Node::op_moment_update2(node->out1 + i * m, node->in1 + i * m,
                                                                node->cons1, node->cons2, m, preds));

            div_nodes.push_back(Node::op_moment_update2(node->out1  + (n - 1) * m, node->in1  + (n - 1) * m,
                                                                node->cons1, node->cons2, node->len1 - (n - 1) * m, preds));

            for (auto n : div_nodes)
                graph.add(n);

            auto res = Node::nop(div_nodes);
            graph.add(res);
            return res;
        }

        Node* opti_adam_update(Graph& graph, Node* node, const std::vector<Node*>& preds)
        {
            std::size_t n;
            std::size_t m;
            elemwhise_size(node->len1, n, m);

            if (n < 2)
            {
                auto res = Node::op_adam_update(node->out1, node->out2,
                                                node->in1, node->in2,
                                                node->cons1, node->cons2,
                                                node->cons3, node->cons4,
                                                node->len1, preds);
                graph.add(res);
                return res;
            }

            std::vector<Node*> div_nodes;
            for (std::size_t i = 0; i < n - 1; ++i)
                div_nodes.push_back(Node::op_adam_update(node->out1 + i * m, node->out2 + i * m,
                                                        node->in1 + i * m, node->in2 + i * m,
                                                        node->cons1, node->cons2,
                                                        node->cons3, node->cons4,
                                                        m, preds));

            div_nodes.push_back(Node::op_adam_update(node->out1 + (n - 1) * m, node->out2 + (n - 1) * m,
                                                        node->in1 + (n - 1) * m, node->in2 + (n - 1) * m,
                                                        node->cons1, node->cons2,
                                                        node->cons3, node->cons4,
                                                        node->len1 - (n - 1) * m, preds));

            for (auto n : div_nodes)
                graph.add(n);

            auto res = Node::nop(div_nodes);
            graph.add(res);
            return res;
        }

        Node* opti_leaky_relu_grad(Graph& graph, Node* node, const std::vector<Node*>& preds)
        {
            std::size_t n;
            std::size_t m;
            elemwhise_size(node->len1, n, m);

            if (n < 2)
            {
                auto res = Node::op_leaky_relu_grad(node->in1, node->in2, node->out1,
                                                    node->cons1, node->len1, preds);
                graph.add(res);
                return res;
            }

            std::vector<Node*> div_nodes;
            for (std::size_t i = 0; i < n - 1; ++i)
                div_nodes.push_back(Node::op_leaky_relu_grad(node->in1 + i * n, node->in2 + i * n,
                                                    node->out1 + i * n, node->cons1, m, preds));

            div_nodes.push_back(Node::op_leaky_relu_grad(node->in1 + (n - 1) * m, node->in2 + (n - 1) * m,
                                                    node->out1 + (n - 1) * m, node->cons1, node->len1 - (n - 1) * m, preds));

            for (auto n : div_nodes)
                graph.add(n);

            auto res = Node::nop(div_nodes);
            graph.add(res);
            return res;
        }

        Node* opti_conv2d_bias_add_grad(Graph& graph, Node* node, const std::vector<Node*>& preds)
        {
            auto res = Node::op_conv2d_bias_add_grad(node->in1, node->sizes1,
                                                     node->out1, preds);
            graph.add(res);
            return res;
        }

        Node* opti_tanh_grad(Graph& graph, Node* node, const std::vector<Node*>& preds)
        {
            std::size_t n;
            std::size_t m;
            elemwhise_size(node->len1, n, m);

            if (n < 2)
            {
                auto res = Node::op_tanh_grad(node->in1, node->in2, node->out1, node->len1, preds);
                graph.add(res);
                return res;
            }

            std::vector<Node*> div_nodes;
            for (std::size_t i = 0; i < n - 1; ++i)
                div_nodes.push_back(Node::op_tanh_grad(node->in1 + i * n, node->in2 + i * n,
                                                    node->out1 + i * n, m, preds));

            div_nodes.push_back(Node::op_tanh_grad(node->in1 + (n - 1) * m, node->in2 + (n - 1) * m,
                                                    node->out1 + (n - 1) * m, node->len1 - (n - 1) * m, preds));

            for (auto n : div_nodes)
                graph.add(n);

            auto res = Node::nop(div_nodes);
            graph.add(res);
            return res;
        }

        Node* opti_conv2d_transpose(Graph& graph, Node* node, const std::vector<Node*>& preds)
        {
            auto res = Node::op_conv2d_transpose(node->in1, node->in2,
                                                 node->sizes1, node->intconst,
                                                 node->out1, node->sizes2,
                                                 node->sizes3, preds);
            graph.add(res);
            return res;
        }

        Node* opti_conv2d_transpose_input_grad(Graph& graph, Node* node,
                                               const std::vector<Node*>& preds)
        {
            auto res = Node::op_conv2d_transpose_input_grad(node->in1, node->in2,
                                                            node->intconst, node->out1,
                                                            node->sizes1, node->sizes2,
                                                            node->intconst2, preds);
            graph.add(res);
            return res;
        }

        Node* opti_conv2d_transpose_kernel_grad(Graph& graph, Node* node,
                                                const std::vector<Node*>& preds)
        {
            auto res = Node::op_conv2d_transpose_kernel_grad(node->in1, node->in2,
                                                             node->intconst, node->out1,
                                                             node->sizes1, node->sizes2,
                                                             node->sizes3,
                                                             preds);
            graph.add(res);
            return res;
        }

        Node* opti_add(Graph& graph, Node* node, const std::vector<Node*>& preds)
        {
            std::size_t n;
            std::size_t m;
            elemwhise_size(node->len1, n, m);

            if (n < 2)
            {
                auto res = Node::op_add(node->in1, node->in2, node->out1,
                                        node->len1, preds);
                //res->use_simd = node->len1 % SIZE_DIVISOR == 0;
                    
                graph.add(res);
                return res;
            }

            const dbl_t* in1 = node->in1;
            const dbl_t* in2 = node->in2;
            dbl_t* out = node->out1;

            std::vector<Node*> div_nodes;
            for (std::size_t i = 0; i < n - 1; ++i)
                div_nodes.push_back(Node::op_add(in1 + i * m, in2 + i * m,out + i * m,
                                                 m, preds));

            div_nodes.push_back(Node::op_add(in1 + (n - 1) * m, in2 + (n - 1) * m, out + (n - 1) * m,
                                             node->len1 - ((n - 1) * m), preds));

            for (auto n : div_nodes)
            {
                //n->use_simd = true;
                graph.add(n);
            }

            auto res = Node::nop(div_nodes);
            graph.add(res);
            return res;
        }


        opti_f optis_list[64] = {
            opti_mat_mat_mul,
            opti_mat_rvect_add,
            opti_sigmoid,
            opti_mse,
            opti_softmax,
            opti_log_softmax,
            opti_softmax_cross_entropy,
            opti_conv2d,
            opti_relu,
            opti_relu_leaky,
            opti_tanh,
            opti_mse_grad,
            opti_sigmoid_grad,
            opti_mat_mul_add,
            opti_tmat_mat_mul,
            opti_mat_tmat_mul,
            opti_mat_sum_rows,
            opti_mat_sum_cols,
            opti_softmax_cross_entropy_grad,
            opti_relu_grad,
            opti_conv2d_bias_add,
            opti_update,
            opti_sigmoid_cross_entropy,
            opti_sigmoid_cross_entropy_grad,
            opti_conv2d_input_grad,
            opti_conv2d_kernel_grad,
            opti_argmax_acc,
            opti_moment_update,
            opti_moment_update2,
            opti_adam_update,
            opti_leaky_relu_grad,
            opti_conv2d_bias_add_grad,
            opti_tanh_grad,
            opti_conv2d_transpose,
            opti_conv2d_transpose_input_grad,
            opti_conv2d_transpose_kernel_grad,
            opti_add
        };
        
    }

    namespace
    {

        Node* opti_node(Node* node, Graph& graph, std::map<Node*, Node*>& optis)
        {
            auto it = optis.find(node);
            if (it != optis.end())
                return it->second;

            std::vector<Node*> preds;
            for (auto n : node->preds)
                preds.push_back(opti_node(n, graph, optis));

            Node* res;
            if (node->type == Node::OP_NOP)
            {
                res = Node::nop(preds);
                graph.add(res);
            }
            else
                res = optis_list[node->type](graph, node, preds);

            optis[node] = res;
            return res;
        }
        
    }

    Graph* optimize(const Graph& graph, std::map<Node*, Node*>& optis)
    {
        optis.clear();
        auto graph_opti = new Graph;
        for (auto node : graph.nodes())
            opti_node(node, *graph_opti, optis);
        return graph_opti;
    }

    std::vector<Node*> convert_nodes(const std::vector<Node*> nodes,
                                     const std::map<Node*, Node*>& optis)
    {
        std::vector<Node*> res;
        for (auto n : nodes)
        {
            auto it = optis.find(n);
            assert(it != optis.end());
            res.push_back(it->second);
        }
        return res;
    }
    
}
