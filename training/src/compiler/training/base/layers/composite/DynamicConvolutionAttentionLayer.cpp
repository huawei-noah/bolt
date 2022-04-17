// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "DynamicConvolutionAttentionLayer.h"
#include "AttentionMaskCreatorLayer.h"
#include "DynamicConvolutionAttentionInternalLayers.h"

#include <training/base/layers/activations/SoftMaxActivation.h>
#include <training/base/layers/activations/TanhActivation.h>
#include <training/base/layers/basic/ArgMaxLayer.h>
#include <training/base/layers/basic/DataLayer.h>
#include <training/base/layers/basic/DynamicDepthwiseConvolution2DLayer.h>
#include <training/base/layers/basic/ElementWiseCompareLayer.h>
#include <training/base/layers/basic/ElementWiseMaxLayer.h>
#include <training/base/layers/basic/ElementWiseMulLayer.h>
#include <training/base/layers/basic/ElementWiseSumLayer.h>
#include <training/base/layers/basic/LogLayer.h>
#include <training/base/layers/basic/PaddingLayer.h>
#include <training/base/layers/basic/ReduceSumLayer.h>
#include <training/base/layers/basic/ReshapeLayer.h>
#include <training/base/layers/basic/SelectLayer.h>
#include <training/base/layers/basic/TransposeLayer.h>
#include <training/base/layers/basic/trainable/Convolution1DLayer.h>
#include <training/base/layers/basic/trainable/LinearLayer.h>

namespace raul
{

DynamicConvolutionAttentionLayer::DynamicConvolutionAttentionLayer(const Name& name, const DynamicConvolutionAttentionParams& params, raul::NetworkParameters& networkParameters)
    : mNumUnits(params.mNumUnits)
    , mCumulativeMode(params.mCumulateWeights)
    , mLocationConvolutionFilters(params.mHparams.mAttentionFilters)
    , mLocationConvolutionKernelSize(params.mHparams.mAttentionKernel)
    , mPriorFilterSize(params.mHparams.mPriorFilterSize)
    , mPriorAlpha(params.mHparams.mPriorAlpha)
    , mPriorBeta(params.mHparams.mPriorBeta)
    , mHasMask(false)
    , mMaskLen(0)
{
    auto prefix = "DynamicConvolutionAttention[" + name + "::ctor]: ";

    // Query, State, Memory, [MemorySeqLength]
    if (params.getInputs().size() != 3 && params.getInputs().size() != 4)
    {
        THROW("DynamicConvolutionAttention", name, "wrong number of input names");
    }

    // Mask independent
    if ((!params.getSharedLayer().empty() && !mCumulativeMode && (params.getOutputs().size() == 3 || params.getOutputs().size() == 4)) ||
        (!params.getSharedLayer().empty() && mCumulativeMode && (params.getOutputs().size() == 1 || params.getOutputs().size() == 4)))
    {
        THROW("DynamicConvolutionAttention", name, "wrong number of output names");
    }

    // With mask
    if (params.getInputs().size() == 4 && params.getSharedLayer().empty() &&
        ((mCumulativeMode && params.getOutputs().size() < 3) || (!mCumulativeMode && params.getOutputs().size() != 2 && params.getOutputs().size() != 3)))
    {
        THROW("DynamicConvolutionAttention", name, "wrong number of output names");
    }

    // Without mask
    if (params.getInputs().size() == 3 && params.getSharedLayer().empty() &&
        ((mCumulativeMode && params.getOutputs().size() != 2 && params.getOutputs().size() != 3) || (!mCumulativeMode && params.getOutputs().size() > 2)))
    {
        THROW("DynamicConvolutionAttention", name, "wrong number of output names");
    }

    // Input names
    auto [queryName, stateName, memoryName] = std::make_tuple(params.getInputs()[0], params.getInputs()[1], params.getInputs()[2]);

    if (params.getInputs().size() == 4 && params.getSharedLayer().empty())
    {
        mHasMask = true;
        mMaskLen = networkParameters.mWorkflow.getWidth(stateName);

        networkParameters.mWorkflow.add<AttentionMaskCreatorLayer>(name / "create_mask", raul::BasicParams{ { params.getInputs()[3] }, { name / "mask" } }, mMaskLen);
    }

    // Trainable params
    if (!params.getSharedLayer().empty())
    {
        mAttentionVPName = params.getSharedLayer() / "attention_variable_projection";
        mAttentionBiasName = params.getSharedLayer() / "attention_bias";
    }
    else
    {
        mAttentionVPName = name / "attention_variable_projection";
        mAttentionBiasName = name / "attention_bias";

        // Also init prior filters
        networkParameters.mWorkflow.add<raul::dca::DCATrainableInitializerLayer>(
            name / "trainable_params", raul::TrainableParams{ {}, { mAttentionVPName, mAttentionBiasName }, params.frozen }, mNumUnits, mPriorAlpha, mPriorBeta, mPriorFilterSize);
    }

    // Calculate values
    if (mHasMask)
    {
        networkParameters.mWorkflow.add<raul::ElementWiseMulLayer>(name / "mask_memory", raul::ElementWiseLayerParams{ { memoryName, name / "mask" }, { params.getOutputs()[1] } });
    }

    // Get keys
    if (params.getSharedLayer().empty())
    {
        // Process memory from [batch, 1, max_time, encoder_output_size] to key with size [batch, 1, max_time, mNumUnits]
        networkParameters.mWorkflow.add<raul::LinearLayer>(name / "memory_layer",
                                                           raul::LinearParams{ { mHasMask ? params.getOutputs()[1] : memoryName }, { name / "keys" }, mNumUnits, false, params.frozen });
    }

    // Transpose state from [batch_size, 1, 1, max_time] to [batch_size, 1, max_time, 1]
    networkParameters.mWorkflow.add<raul::TransposeLayer>(name / "transpose_state", raul::TransposingParams{ { stateName }, { name / "expanded_alignments" }, Dimension::Width, Dimension::Height });

    // Extract location-based features from state. Padding mode is "same"
    size_t locationConvolutionPadding = (mLocationConvolutionKernelSize - 1) / 2;

    networkParameters.mWorkflow.add<raul::Convolution1DLayer>(name / "location_convolution",
                                                              raul::Convolution1DParams{ name / "expanded_alignments",
                                                                                         name / "f",
                                                                                         params.getSharedLayer().empty() ? "" : params.getSharedLayer() / "location_convolution",
                                                                                         mLocationConvolutionKernelSize,
                                                                                         mLocationConvolutionFilters,
                                                                                         1,
                                                                                         locationConvolutionPadding,
                                                                                         1,
                                                                                         1,
                                                                                         true,
                                                                                         false,
                                                                                         true,
                                                                                         params.frozen });

    // Projected location features [batch_size, 1, max_time, attention_dim]
    networkParameters.mWorkflow.add<raul::LinearLayer>(
        name / "location_layer",
        raul::LinearParams{
            { name / "f" }, { name / "processed_location_features" }, params.getSharedLayer().empty() ? "" : params.getSharedLayer() / "location_layer", mNumUnits, false, params.frozen });

    // Get dynamic filters
    networkParameters.mWorkflow.add<raul::LinearLayer>(
        name / "dynamic_fc1",
        raul::LinearParams{ { queryName }, { name / "intermediate_dynamic_filters" }, params.getSharedLayer().empty() ? "" : params.getSharedLayer() / "dynamic_fc1", 128, true, params.frozen });

    // Tanh activation
    networkParameters.mWorkflow.add<raul::TanhActivation>(name / "dynamic_fc1_activation",
                                                          raul::BasicParams{ { name / "intermediate_dynamic_filters" }, { name / "activated_intermediate_dynamic_filters" } });

    networkParameters.mWorkflow.add<raul::LinearLayer>(name / "dynamic_fc2",
                                                       raul::LinearParams{ { name / "activated_intermediate_dynamic_filters" },
                                                                           { name / "dynamic_filters" },
                                                                           params.getSharedLayer().empty() ? "" : params.getSharedLayer() / "dynamic_fc2",
                                                                           168,
                                                                           false,
                                                                           params.frozen });

    // Final filters for dynamic convolution
    networkParameters.mWorkflow.add<raul::ReshapeLayer>(name / "reshape_filters", raul::ViewParams{ name / "dynamic_filters", name / "dynamic_filters_reshaped", 1u, 21u, 8u });

    // Transpose filters [0, 1, 2, 3] -> [0, 2, 1, 3]
    networkParameters.mWorkflow.add<raul::TransposeLayer>(
        name / "transpose_filters_step_1", raul::TransposingParams{ { name / "dynamic_filters_reshaped" }, { name / "dynamic_filters_transposed" }, Dimension::Depth, Dimension::Height });

    // Transpose filters [0, 2, 1, 3] -> [1, 2, 0, 3]
    networkParameters.mWorkflow.add<raul::TransposeLayer>(name / "transpose_filters_step_2",
                                                          raul::TransposingParams{ { name / "dynamic_filters_transposed" }, { name / "final_dynamic_filters" }, Dimension::Batch, Dimension::Height });

    // Begin of the section which is not as in reference python code!

    // Pad input for dynamic convolution
    networkParameters.mWorkflow.add<raul::PaddingLayer>(name / "pad_input_for_dc",
                                                        raul::PaddingLayerParams{ { name / "expanded_alignments" }, { name / "pre_dynamic_input" }, 10u, 10u, 0u, 0u, 0.0_dt });

    // Transpose input for dynamic convolution - simplified! (additionally check it)
    networkParameters.mWorkflow.add<raul::TransposeLayer>(name / "transpose_input_for_dc",
                                                          raul::TransposingParams{ { name / "pre_dynamic_input" }, { name / "dynamic_input" }, Dimension::Batch, Dimension::Width });

    // End of this section

    // Dynamic convolution
    networkParameters.mWorkflow.add<raul::DynamicDepthwiseConvolution2DLayer>(name / "dynamic_convolution",
                                                                              raul::BasicParams{ { name / "dynamic_input", name / "final_dynamic_filters" }, { name / "dynamic_features_1" } });

    // Reshape calculated features
    networkParameters.mWorkflow.add<raul::dca::CustomReshapeLayer>(name / "get_final_dynamic_features", raul::BasicParams{ { name / "dynamic_features_1" }, { name / "dynamic_features" } });

    // Project obtained features
    networkParameters.mWorkflow.add<raul::LinearLayer>(name / "dynamic_projection",
                                                       raul::LinearParams{ { name / "dynamic_features" },
                                                                           { name / "processed_dynamic_features" },
                                                                           params.getSharedLayer().empty() ? "" : params.getSharedLayer() / "dynamic_projection",
                                                                           mNumUnits,
                                                                           false,
                                                                           params.frozen });

    // Padded expanded state
    networkParameters.mWorkflow.add<raul::PaddingLayer>(name / "pad_state",
                                                        raul::PaddingLayerParams{ { name / "expanded_alignments" }, { name / "padded_expanded_alignments" }, 10u, 0u, 0u, 0u, 0.0_dt });

    // Apply prior filters
    // prior_filters - constant tensor, so layer is always frozen
    networkParameters.mWorkflow.add<raul::Convolution1DLayer>(name / "apply_prior_filters",
                                                              raul::Convolution1DParams{ name / "padded_expanded_alignments",
                                                                                         name / "prior",
                                                                                         params.getSharedLayer().empty() ? "" : params.getSharedLayer() / "apply_prior_filters",
                                                                                         mPriorFilterSize,
                                                                                         1,
                                                                                         1,
                                                                                         0,
                                                                                         1,
                                                                                         1,
                                                                                         true,
                                                                                         false,
                                                                                         true,
                                                                                         true });

    networkParameters.mWorkflow.add<raul::dca::DCAConstantsInitializerLayer>(name / "constants", raul::BasicParams{ {}, { name / "MIN_INPUT", name / "MIN_OUTPUT" } });

    networkParameters.mWorkflow.add<raul::ElementWiseMaxLayer>(name / "limit_prior", raul::ElementWiseLayerParams{ { name / "prior", name / "MIN_INPUT" }, { name / "prior_limited" } });

    networkParameters.mWorkflow.add<raul::LogLayer>(name / "log_prior", raul::ElementWiseLayerParams{ { name / "prior_limited" }, { name / "prior_logged" } });

    // Again exclude small values
    networkParameters.mWorkflow.add<raul::ElementWiseCompareLayer>(name / "mask_prior_output",
                                                                   raul::ElementWiseComparisonLayerParams{ { name / "prior", name / "MIN_INPUT" }, { name / "prior_mask" }, true, "ge", 0.0_dt });

    networkParameters.mWorkflow.add<raul::SelectLayer>(name / "generate_prior_output",
                                                       raul::ElementWiseLayerParams{ { name / "prior_mask", name / "prior_logged", name / "MIN_OUTPUT" }, { name / "prior_output" } });

    // Calculate DCA score (Bahdanau-style)
    networkParameters.mWorkflow.add<raul::ElementWiseSumLayer>(
        name / "sum_location_and_dynamic_features",
        raul::ElementWiseLayerParams{ { name / "processed_location_features", name / "processed_dynamic_features", mAttentionBiasName }, { name / "sum_of_features" } });

    networkParameters.mWorkflow.add<raul::TanhActivation>(name / "activate_sum", raul::BasicParams{ { name / "sum_of_features" }, { name / "act_sum_of_features" } });

    networkParameters.mWorkflow.add<raul::ElementWiseMulLayer>(name / "multibly_by_attention_vp",
                                                               raul::ElementWiseLayerParams{ { name / "act_sum_of_features", mAttentionVPName }, { name / "increased_act_sum_of_features" } });

    networkParameters.mWorkflow.add<raul::ReduceSumLayer>(name / "reduce_to_energy",
                                                          raul::BasicParamsWithDim{ { name / "increased_act_sum_of_features" }, { name / "pre_energy" }, raul::Dimension::Width });

    networkParameters.mWorkflow.add<raul::ElementWiseSumLayer>(name / "calculate_energy", raul::ElementWiseLayerParams{ { name / "pre_energy", name / "prior_output" }, { name / "energy" } });

    // Transpose
    networkParameters.mWorkflow.add<raul::TransposeLayer>(name / "transpose_energy",
                                                          raul::TransposingParams{ { name / "energy" }, { name / "transposed_energy" }, Dimension::Width, Dimension::Height });

    networkParameters.mWorkflow.add<raul::SoftMaxActivation>(name / "activate_energy", raul::BasicParamsWithDim{ { name / "transposed_energy" }, { params.getOutputs()[0] }, Dimension::Width });

    // Get new alignments
    if (mCumulativeMode)
    {
        const size_t i = mHasMask ? 2 : 1;
        // Get new state
        networkParameters.mWorkflow.add<raul::ElementWiseSumLayer>(name / "calculate_new_state", raul::ElementWiseLayerParams{ { params.getOutputs()[0], stateName }, { params.getOutputs()[i] } });
    }

    if (params.getOutputs().size() == 4 || ((!mHasMask || !mCumulativeMode) && params.getOutputs().size() == 3) || (!mHasMask && !mCumulativeMode && params.getOutputs().size() == 2))
    {
        // Get max attention indices
        networkParameters.mWorkflow.add<raul::ArgMaxLayer>(name / "get_max_attn", raul::BasicParamsWithDim{ { params.getOutputs()[0] }, { params.getOutputs().back() }, raul::Dimension::Width });
    }
}

}