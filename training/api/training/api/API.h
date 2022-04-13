// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_TRAINING_API
#define _H_TRAINING_API

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

// Losses
#define BINARY_CROSS_ENTROPY_LOSS "OpBinaryCrossEntropyLoss"
#define CROSS_ENTROPY_LOSS "OpCrossEntropyLoss"
#define KL_DIV_LOSS "OpKLDivLoss"
#define L1_LOSS "OpL1Loss"
#define MSE_LOSS "OpMSELoss"
#define NLL_LOSS "OpNLLLoss"
#define SIGMOID_CROSS_ENTROPY_LOSS "OpSigmoidCrossEntropyLoss"
#define SOFTMAX_CROSS_ENTROPY_LOSS "OpSoftmaxCrossEntropyLoss"
#define WEIGHTED_LOSS "OpWeightedLoss"

// Layers
#define CONVOLUTION_2D_LAYER "OpConv2DLayer"
#define CONVOLUTION_DEPTHWISE_2D_LAYER "OpConvDW2DLayer"
#define MAX_POOLING_2D_LAYER "OpMaxPool2DLayer"
#define AVERAGE_POOLING_2D_LAYER "OpAvgPool2DLayer"
#define ELEMENTWISE_DIV_LAYER "OpElementWiseDivLayer"
#define ELEMENTWISE_MAX_LAYER "OpElementWiseMaxLayer"
#define ELEMENTWISE_MIN_LAYER "OpElementWiseMinLayer"
#define ELEMENTWISE_MUL_LAYER "OpElementWiseMulLayer"
#define ELEMENTWISE_SUB_LAYER "OpElementWiseSubLayer"
#define ELEMENTWISE_SUM_LAYER "OpElementWiseSumLayer"
#define ELEMENTWISE_COMPARE_LAYER "OpElementWiseCompareLayer"

// Activations
#define GELU_ERF_ACTIVATION "OpGeLUErfActivation"
#define GELU_TANH_ACTIVATION "OpGeLUTanhActivation"
#define HSIGMOID_ACTIVATION "OpHSigmoidActivation"
#define HSWISH_ACTIVATION "OpHSwishActivation"
#define LOG_SOFTMAX_ACTIVATION "OpLogSoftmaxActivation"
#define RELU_ACTIVATION "OpReLUActivation"
#define RELU6_ACTIVATION "OpReLU6Activation"
#define SIGMOID_ACTIVATION "OpSigmoidActivation"
#define SOFTMAX_ACTIVATION "OpSoftmaxActivation"
#define SWISH_ACTIVATION "OpSwishActivation"
#define TANH_ACTIVATION "OpTanhActivation"

#ifdef __cplusplus
extern "C"
{
#endif

#if defined(_MSC_VER)
#ifdef API_EXPORTS
#define API_EXPORT __declspec(dllexport)
#else
#define API_EXPORT __declspec(dllimport)
#endif
#else
#define API_EXPORT
#endif

#define FLOAT_TYPE float

    enum API_STATUS
    {
        STATUS_OK = 0,
        STATUS_NOT_IMPLEMENTED = 50,
        STATUS_ERROR = 61,
        STATUS_ERROR_BAD_SIZE = 62,
        STATUS_ERROR_BAD_NAME = 63
    };

    enum COMPARISON_TYPE
    {
        COMPARE_EQUAL,
        COMPARE_NOT_EQUAL,
        COMPARE_LESS,
        COMPARE_NOT_LESS,
        COMPARE_GREATER,
        COMPARE_NOT_GREATER
    };

    enum LOSS_REDUCTION
    {
        LOSS_REDUCTION_MEAN = 0,
        LOSS_REDUCTION_SUM = 1,
        LOSS_REDUCTION_BATCH_MEAN = 2
    };

    enum DIM
    {
        DIM_DEFAULT = -1,
        DIM_BATCH = 0,
        DIM_DEPTH = 1,
        DIM_HEIGHT = 2,
        DIM_WIDTH = 3
    };

    enum FILLING_MODE
    {
        CONSTANT,
        REFLECTION,
        REPLICATION
    };

    enum CONSTRAINT_TYPE
    {
        CPU,
        CPUFP16,
        CPUFP16FP32MasterWeights,
        CPUFP32FP16MixedLocal
    };
    /**
     * @brief Graph_Description_t represents structure of neural network
     */
    struct Graph_Description_t;

    /**
     * @brief Graph_t represents complete graph
     */
    struct Graph_t;

    /**
     * @brief Optimizer_t represents an optimizer for network training (SGD, Adam etc.)
     */
    struct Optimizer_t;

    /**
     * @brief Initializer_t represents an initializer for network training (Random, Constant, Xavier etc.)
     */
    struct Initializer_t;

    /**
     * @brief LrScheduler_t represents a lerning rate scheduler for network training
     */
    struct LrScheduler_t;

    API_EXPORT API_STATUS create_graph_description(Graph_Description_t** descr);
    API_EXPORT API_STATUS create_graph_description_fp16(Graph_Description_t** descr);
    API_EXPORT API_STATUS create_graph_description_eager(Graph_Description_t** descr);
    API_EXPORT API_STATUS create_graph_description_compiler(Graph_Description_t** descr);
    API_EXPORT API_STATUS create_graph_description_compiler_fp16(Graph_Description_t** descr);


    API_EXPORT API_STATUS delete_graph_description(Graph_Description_t* descr);

    API_EXPORT API_STATUS create_graph(Graph_Description_t** desc, Graph_t** graph, size_t batch_size);
    API_EXPORT API_STATUS create_graph_with_data_grads(Graph_Description_t** desc, Graph_t** graph, size_t batch_size);

    API_EXPORT API_STATUS delete_graph(Graph_t* graph);

    API_EXPORT const char* get_last_error();

    /**
     * @brief Internal usage only
     */
    API_EXPORT void set_last_error(const char*);

    // optimizers

    API_EXPORT API_STATUS create_adadelta_optimizer(Optimizer_t** optimizer, FLOAT_TYPE learning_rate);
    API_EXPORT API_STATUS create_adagrad_optimizer(Optimizer_t** optimizer, FLOAT_TYPE learning_rate);
    /**
     * @brief Adam (Adaptive moment estimation)
     *
     *  The Adam method computes individual adaptive learning rates for
     *  different parameters from estimates of first
     *  and second moments of the gradients. This method is combination
     *  of AdaGrad and RMSProp. AdaGrad works well with sparse gradients.
     *  RMSProp works well in on-line and non-stationary settings.
     *
     *  \f[
     *      m_t =  \beta_1 m_{t-1} - (1-\beta_1) \nabla_{\theta} E(\theta_{t-1}),\\
     *      \nu_t =  \beta_2 \nu_{t-1} - (1-\beta_2) \nabla^2_{\theta} E(\theta_{t-1}),\\
     *      \hat m_t = \frac{m}{1-\beta_1^t}, \\
     *      \hat \nu_t = \frac{\nu}{1-\beta_2^t}, \\
     *      \theta_{t} =  \theta_{t-1} - \alpha \frac{m_{t}}{\sqrt{\hat \nu_t} + \epsilon},
     *  \f]
     *  where
     *  - \f$m\f$ is the 1st moment vector (the mean of gradient),
     *  - \f$\nu\f$ is the 2st moment vector (the uncentered variance of gradient),
     *  - \f$\beta_1\f$ is the exponential decay rate for 1st moment,
     *  - \f$\beta_2\f$ is the exponential decay rate for 2st moment,
     *  - \f$\hat m\f$ is the bias-corrected 1st moment vector,
     *  - \f$\hat \nu\f$ is the bias-corrected 2st moment vector,
     *  - \f$\theta\f$ is a tuned parameter at specific step of the algorithm,
     *  - \f$\alpha\f$ is a learning rate,
     *  - \f$E(\theta)\f$ is an objective function (error function in our case).
     *
     *  Good default settings from the original article:
     *  - \f$\alpha = 0.0001\f$
     *  - \f$\beta_1 = 0.9\f$
     *  - \f$\beta_2 = 0.999\f$
     *  - \f$\epsilon = 10^{-8}\f$
     *
     *  @see
     *  - D. P. Kingma and J. Ba, �Adam: A Method for Stochastic Optimization� arXiv:1412.6980 [cs], Jan. 2017.
     */
    API_EXPORT API_STATUS create_adam_optimizer(Optimizer_t** optimizer, FLOAT_TYPE alfa, FLOAT_TYPE beta_1, FLOAT_TYPE beta_2, FLOAT_TYPE epsilon);
    API_EXPORT API_STATUS create_adamax_optimizer(Optimizer_t** optimizer, FLOAT_TYPE learning_rate);
    /**
     * @brief Momentum method
     *
     *  The momentum method is a technique for accelerating
     *  gradient descent that accumulates a velocity
     *  vector in directions of persistent reduction in the
     *  objective across iterations.
     *
     *  \f[
     *      \nu_{t} =  \mu nu_{t-1} - \eta_{t-1} \nabla_{\theta} E(\theta_{t-1}),\\
     *      \theta_{t} =  \theta_{t-1} - \nu_{t}
     *  \f]
     *  where
     *  - \f$\nu\f$ is a velocity,
     *  - \f$\mu\f$ is a momentum parameter,
     *  - \f$\theta\f$ is a tuned parameter at specific step of the algorithm,
     *  - \f$\eta\f$ is a learning rate,
     *  - \f$E(\theta)\f$ is an objective function (error function in our case).
     *
     *  @see
     *  - I. Sutskever, J. Martens, G. Dahl, and G. Hinton, �On the importance of initialization and momentum in deep learning� p. 14.
     */
    API_EXPORT API_STATUS create_momentum_optimizer(Optimizer_t** optimizer, FLOAT_TYPE learning_rate, FLOAT_TYPE momentum);
    API_EXPORT API_STATUS create_nesterov_optimizer(Optimizer_t** optimizer, FLOAT_TYPE learning_rate, FLOAT_TYPE momentum);
    /**
     * @brief Stochastic gradient descent (SGD)
     *
     *  This is classical stochastic gradient descent with
     *  one parameter: learning rate (lr). An optimization
     *  algorithm works according to the following formula.
     *
     *  \f[
     *      \theta_{t} =  \theta_{t-1} - \eta_{t-1} \nabla_{\theta} E(\theta_{t-1}),
     *  \f]
     *  where
     *  - \f$\theta\f$ is a tuned parameter at specific step of the algorithm,
     *  - \f$\eta\f$ is a learning rate,
     *  - \f$E(\theta)\f$ is an objective function (error function in our case).
     *
     *  @see
     *  - S. Ruder, �An overview of gradient descent optimization algorithms� arXiv:1609.04747 [cs], Jun. 2017.
     */
    API_EXPORT API_STATUS create_sgd_optimizer(Optimizer_t** optimizer, FLOAT_TYPE learning_rate);
    API_EXPORT API_STATUS delete_optimizer(Optimizer_t* optimizer);

    // initializers
    API_EXPORT API_STATUS create_constant_initializer(Initializer_t** initializer, FLOAT_TYPE value);
    API_EXPORT API_STATUS create_random_norm_initializer(Initializer_t** initializer, FLOAT_TYPE mean, FLOAT_TYPE stddev, size_t seed);
    API_EXPORT API_STATUS create_random_uniform_initializer(Initializer_t** initializer, FLOAT_TYPE min_value, FLOAT_TYPE max_value, size_t seed);
    API_EXPORT API_STATUS create_xavier_norm_initializer(Initializer_t** initializer, size_t seed);
    API_EXPORT API_STATUS create_xavier_uniform_initializer(Initializer_t** initializer, size_t seed);
    API_EXPORT API_STATUS delete_initializer(Initializer_t* initializer);

    API_EXPORT API_STATUS initialize_tensor(Graph_t* graph, Initializer_t* initializer, const char* tensor_name);
    API_EXPORT API_STATUS arange(Graph_t* graph, const char* tensor_name, FLOAT_TYPE start, FLOAT_TYPE step);

    // scheduler
    API_EXPORT API_STATUS create_cosine_annealing_adam_w_lr_scheduler(LrScheduler_t** scheduler, size_t size, FLOAT_TYPE max_a, FLOAT_TYPE min_a, FLOAT_TYPE warmup_percentage, FLOAT_TYPE warmup_pow, FLOAT_TYPE annealing_pow, FLOAT_TYPE base_lr, FLOAT_TYPE beta_1, FLOAT_TYPE beta_2, FLOAT_TYPE epsilon, FLOAT_TYPE weight_decay);

    // training
    API_EXPORT API_STATUS set_batch_size(Graph_t* graph, size_t batch_size);

    /**
     * Performs one iteration of training
     * @param loss_names same as loss_name in add_loss_layer
     * @param loss array of size loss_count
     */
    API_EXPORT API_STATUS train_single_pass(Graph_t* graph, Optimizer_t* optimizer, const char** loss_names, size_t loss_count, FLOAT_TYPE* loss);
    API_EXPORT API_STATUS train_single_pass_with_scheduling(Graph_t* graph, LrScheduler_t* scheduler, const char** loss_names, size_t loss_count, FLOAT_TYPE* loss);
    API_EXPORT API_STATUS test_network(Graph_t* graph, const char* prob_tensor_name, size_t* correctClasses, FLOAT_TYPE* accuracy);
    API_EXPORT API_STATUS get_model_parameters(Graph_t* graph, bool only_trainable, char** parameters, size_t* param_count, size_t* max_param_name_length);

    // inference for verification
    API_EXPORT API_STATUS network_forward(Graph_t* graph, bool is_test);
    API_EXPORT API_STATUS network_backward(Graph_t* graph);

    API_EXPORT API_STATUS
    add_data_layer_with_labels(Graph_Description_t* desc, const char* name, const char** output_names, size_t output_count, size_t depth, size_t height, size_t width, size_t labels_count);
    API_EXPORT API_STATUS add_data_layer(Graph_Description_t* desc, const char* name, const char** output_names, size_t output_count, size_t depth, size_t height, size_t width);

    API_EXPORT API_STATUS add_embedding_layer(Graph_Description_t* desc,
                                              const char* name,
                                              const char* input_name,
                                              const char* output_name,
                                              size_t dictionary_size,
                                              size_t embedding_size,
                                              int padding_idx,
                                              bool scale_by_size,
                                              bool scale_grad_by_frequency);

    API_EXPORT API_STATUS add_labels(Graph_Description_t* desc, const char* name, const char** output_names, size_t output_count, size_t labels_count);

    API_EXPORT API_STATUS
    add_transpose_layer(Graph_Description_t* desc, const char* name, const char* input_name, const char* output_name, DIM from, DIM to);

    API_EXPORT API_STATUS
    add_reshape_layer(Graph_Description_t* desc, const char* name, const char* input_name, const char* output_name, int new_depth, int new_height, int new_width);

    API_EXPORT API_STATUS
    add_loss_layer(Graph_Description_t* desc, const char* name, const char** input_names, const char* loss_name, const char* loss_type, size_t inputs_count, LOSS_REDUCTION reduction);

    API_EXPORT API_STATUS
    add_loss_layer_with_compiler(Graph_Description_t* desc, const char* name, const char** input_names, const char* loss_name, const char* loss_type, size_t inputs_count, LOSS_REDUCTION reduction);

    API_EXPORT API_STATUS add_lstm_layer(Graph_Description_t* desc, const char* name, const char* input_name, const char* output_name, size_t hidden_size, bool use_global_fusion, bool use_bias);
    API_EXPORT API_STATUS add_lstm_layer_ext(Graph_Description_t* desc,
                                             const char* name,
                                             const char* input_name,
                                             const char* hidden_input_name,
                                             const char* cell_input_name,
                                             const char* output_name,
                                             const char* hidden_output_name,
                                             const char* cell_output_name,
                                             bool use_global_fusion,
                                             bool use_bias);

    API_EXPORT API_STATUS print_graph(Graph_t* graph);
    API_EXPORT API_STATUS print_graph_to_file(Graph_t* graph, FILE* file);
    API_EXPORT API_STATUS print_graph_to_string(Graph_t* graph, char* string, size_t* length);

    API_EXPORT API_STATUS get_tensor(Graph_t* graph, const char* tensor_name, FLOAT_TYPE* data, size_t* size);
    API_EXPORT API_STATUS fill_tensor(Graph_t* graph, const char* tensor_name, const FLOAT_TYPE value);
    API_EXPORT API_STATUS set_tensor(Graph_t* graph, const char* tensor_name, const FLOAT_TYPE* data, size_t size);
    API_EXPORT API_STATUS create_tensor(Graph_t* graph, const char* tensor_name, size_t batchSize, size_t depth, size_t height, size_t width);

    API_EXPORT API_STATUS set_constraint(Graph_Description_t* desc, const char* layer_name, CONSTRAINT_TYPE constraint);
    API_EXPORT API_STATUS set_constraint_sequence(Graph_Description_t* desc, const char* layer_name_from, const char* layer_name_to, CONSTRAINT_TYPE constraint);

    API_EXPORT API_STATUS reset_layer_execution_target_override(Graph_Description_t* desc);
    API_EXPORT API_STATUS finetune(Graph_t* graph, LrScheduler_t* lr_scheduler, const char** input_names, const FLOAT_TYPE** input_data, const int* input_sizes, const int input_num, const char* loss_name, size_t step);
#ifdef __cplusplus
}
#endif

#endif
