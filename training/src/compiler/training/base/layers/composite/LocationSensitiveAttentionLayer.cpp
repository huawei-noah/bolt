// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "LocationSensitiveAttentionLayer.h"
#include "AttentionMaskCreatorLayer.h"
#include "LocationSensitiveAttentionInternalLayers.h"

#include <training/base/layers/activations/SigmoidActivation.h>
#include <training/base/layers/activations/SoftMaxActivation.h>
#include <training/base/layers/activations/TanhActivation.h>
#include <training/base/layers/basic/ArgMaxLayer.h>
#include <training/base/layers/basic/ConcatenationLayer.h>
#include <training/base/layers/basic/ElementWiseDivLayer.h>
#include <training/base/layers/basic/ElementWiseMulLayer.h>
#include <training/base/layers/basic/ElementWiseSubLayer.h>
#include <training/base/layers/basic/ElementWiseSumLayer.h>
#include <training/base/layers/basic/MatMulLayer.h>
#include <training/base/layers/basic/RandomTensorLayer.h>
#include <training/base/layers/basic/ReduceSumLayer.h>
#include <training/base/layers/basic/RollLayer.h>
#include <training/base/layers/basic/ScaleLayer.h>
#include <training/base/layers/basic/SelectLayer.h>
#include <training/base/layers/basic/SplitterLayer.h>
#include <training/base/layers/basic/TensorLayer.h>
#include <training/base/layers/basic/TransposeLayer.h>
#include <training/base/layers/basic/trainable/Convolution1DLayer.h>
#include <training/base/layers/basic/trainable/LinearLayer.h>

namespace raul
{

LocationSensitiveAttentionLayer::LocationSensitiveAttentionLayer(const Name& name, const LocationSensitiveAttentionParams& params, NetworkParameters& networkParameters)
    : mNumUnits(params.mNumUnits)
    , mCumulativeMode(params.mCumulateWeights)
    , mLocationConvolutionFilters(params.mHparams.mAttentionFilters)
    , mLocationConvolutionKernelSize(params.mHparams.mAttentionKernel)
    , mHasMask(false)
    , mTransitionProba(0.5_dt)
{
    auto prefix = "LocationSensitiveAttention[" + name + "::ctor]: ";

    // Query, State, Memory, [MemorySeqLength]
    if (params.getInputs().size() != 3 && params.getInputs().size() != 4)
    {
        THROW("LocationSensitiveAttention", name, "wrong number of input names");
    }

    // Mask independent
    if ((!params.getSharedLayer().empty() && !mCumulativeMode && (params.getOutputs().size() == 3 || params.getOutputs().size() == 4)) ||
        (!params.getSharedLayer().empty() && mCumulativeMode && (params.getOutputs().size() == 1 || params.getOutputs().size() == 4)))
    {
        THROW("LocationSensitiveAttention", name, "wrong number of output names");
    }

    // With mask
    if (params.getInputs().size() == 4 && params.getSharedLayer().empty() &&
        ((mCumulativeMode && params.getOutputs().size() < 3) || (!mCumulativeMode && params.getOutputs().size() != 2 && params.getOutputs().size() != 3)))
    {
        THROW("LocationSensitiveAttention", name, "wrong number of output names");
    }

    // Without mask
    if (params.getInputs().size() == 3 && params.getSharedLayer().empty() &&
        ((mCumulativeMode && params.getOutputs().size() != 2 && params.getOutputs().size() != 3) || (!mCumulativeMode && params.getOutputs().size() > 2)))
    {
        THROW("LocationSensitiveAttention", name, "wrong number of output names");
    }

    // Input names
    auto [queryName, stateName, memoryName] = std::make_tuple(params.getInputs()[0], params.getInputs()[1], params.getInputs()[2]);

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

        networkParameters.mWorkflow.add<lsa::LSATrainableInitializerLayer>(name / "trainable_params", TrainableParams{ {}, { mAttentionVPName, mAttentionBiasName }, params.frozen }, mNumUnits);
    }

    if (params.getInputs().size() == 4 && params.getSharedLayer().empty())
    {
        mHasMask = true;
        mMaskLen = networkParameters.mWorkflow.getWidth(stateName);

        networkParameters.mWorkflow.add<AttentionMaskCreatorLayer>(name / "create_mask", BasicParams{ { params.getInputs()[3] }, { name / "mask", name / "default_scores" } }, mMaskLen);

        networkParameters.mWorkflow.add<ElementWiseMulLayer>(name / "mask_memory", ElementWiseLayerParams{ { memoryName, name / "mask" }, { name / "values" } });
        networkParameters.mWorkflow.add<SplitterLayer>(name / "copy_values_to_output", BasicParams{ { name / "values" }, { params.getOutputs()[1] } });
    }

    // Get keys
    if (params.getSharedLayer().empty())
    {
        // Process memory from [batch, 1, max_time, encoder_output_size] to key with size [batch, 1, max_time, mNumUnits]
        networkParameters.mWorkflow.add<LinearLayer>(name / "memory_layer", LinearParams{ { mHasMask ? params.getOutputs()[1] : memoryName }, { name / "keys" }, mNumUnits, false, params.frozen });
    }

    // Recalculate transition proba if needed
    if (params.mHparams.mUseTransAgent)
    {
        networkParameters.mWorkflow.add<MatMulLayer>(
            name / "calculate_previous_context", MatMulParams{ { stateName, params.getSharedLayer().empty() ? name / "values" : params.getSharedLayer() / "values" }, { name / "previous_context" } });
        networkParameters.mWorkflow.add<ConcatenationLayer>(name / "calculate_ta_input", BasicParamsWithDim{ { queryName, name / "previous_context" }, { name / "ta_input" }, Dimension::Width });
        // Transition layer itself
        networkParameters.mWorkflow.add<LinearLayer>(
            name / "transition_agent_layer",
            LinearParams{ { name / "ta_input" }, { name / "ta_output" }, params.getSharedLayer().empty() ? "" : params.getSharedLayer() / "transition_agent_layer", 1u, true, params.frozen });
        // Get new probability
        networkParameters.mWorkflow.add<SigmoidActivation>(name / "calculate_tr_proba", BasicParams{ { name / "ta_output" }, { name / "transit_proba" } });
    }
    else
    {
        // Use hardcoded value
        networkParameters.mWorkflow.add<TensorLayer>(name / "create_transit_proba", TensorParams{ { name / "transit_proba" }, WShape{ BS(), 1u, 1u, 1u }, mTransitionProba });
    }

    // Process query from [batch, 1, 1, query_depth] to [batch, 1, 1, attention_dim]
    networkParameters.mWorkflow.add<LinearLayer>(
        name / "query_layer",
        LinearParams{ { queryName }, { name / "processed_query" }, params.getSharedLayer().empty() ? "" : params.getSharedLayer() / "query_layer", mNumUnits, false, params.frozen });

    // Transpose state from [batch_size, 1, 1, max_time] to [batch_size, 1, max_time, 1]
    networkParameters.mWorkflow.add<TransposeLayer>(name / "transpose_state", TransposingParams{ { stateName }, { name / "expanded_alignments" }, Dimension::Width, Dimension::Height });

    // Extract location-based features from previous attention. Padding mode is "same"
    size_t locationConvolutionPadding = (mLocationConvolutionKernelSize - 1) / 2;

    networkParameters.mWorkflow.add<Convolution1DLayer>(name / "location_convolution",
                                                        Convolution1DParams{ name / "expanded_alignments",
                                                                             name / "f",
                                                                             params.getSharedLayer().empty() ? "" : params.getSharedLayer() / "location_convolution",
                                                                             mLocationConvolutionKernelSize,
                                                                             mLocationConvolutionFilters,
                                                                             1u,
                                                                             locationConvolutionPadding,
                                                                             1u,
                                                                             1u,
                                                                             true,
                                                                             false,
                                                                             true,
                                                                             params.frozen });

    networkParameters.mWorkflow.add<LinearLayer>(
        name / "location_layer",
        LinearParams{ { name / "f" }, { name / "processed_location_features" }, params.getSharedLayer().empty() ? "" : params.getSharedLayer() / "location_layer", mNumUnits, false, params.frozen });

    // Compute energy
    networkParameters.mWorkflow.add<ElementWiseSumLayer>(
        name / "sum_features",
        ElementWiseLayerParams{
            { name / "processed_query", name / "processed_location_features", params.getSharedLayer().empty() ? name / "keys" : params.getSharedLayer() / "keys", mAttentionBiasName },
            { name / "sum_of_features" } });

    networkParameters.mWorkflow.add<TanhActivation>(name / "activate_features", BasicParams{ { name / "sum_of_features" }, { name / "act_sum_of_features" } });

    networkParameters.mWorkflow.add<ElementWiseMulLayer>(name / "mul_with_attention_vp", ElementWiseLayerParams{ { name / "act_sum_of_features", mAttentionVPName }, { name / "non_reduced_energy" } });

    networkParameters.mWorkflow.add<ReduceSumLayer>(name / "location_sensitive_score", BasicParamsWithDim{ { name / "non_reduced_energy" }, { name / "energy" }, Dimension::Width });

    Name energyName = name / "energy";
    if (params.getInputs().size() == 4 && !params.mHparams.mUseStepwiseMonotonicConstraintType)
    {
        networkParameters.mWorkflow.add<SelectLayer>(name / "mask_energy",
                                                     ElementWiseLayerParams{ { params.getSharedLayer().empty() ? name / "mask" : params.getSharedLayer() / "mask",
                                                                               energyName,
                                                                               params.getSharedLayer().empty() ? name / "default_scores" : params.getSharedLayer() / "default_scores" },
                                                                             { name / "maskedEnergy" } });
        energyName = name / "maskedEnergy";
    }
    // Compute alignments
    networkParameters.mWorkflow.add<TransposeLayer>(name / "transpose_energy", TransposingParams{ { energyName }, { name / "transposed_energy" }, Dimension::Width, Dimension::Height });

    Name alignmentsName = params.mUseForward ? name / "prefinal_alignments" : params.getOutputs()[0];
    if (params.mHparams.mUseStepwiseMonotonicConstraintType)
    {
        networkParameters.mWorkflow.add<RandomTensorLayer>(name / "create_noise",
                                                           RandomTensorLayerParams{ { name / "noise" },
                                                                                    networkParameters.mWorkflow.getDepth(name / "transposed_energy"),
                                                                                    networkParameters.mWorkflow.getHeight(name / "transposed_energy"),
                                                                                    networkParameters.mWorkflow.getWidth(name / "transposed_energy") });
        networkParameters.mWorkflow.add<ScaleLayer>(name / "scale_noise", ScaleParams{ { name / "noise" }, { name / "scaled_noise" }, params.mSigmoidNoise });
        networkParameters.mWorkflow.add<ElementWiseSumLayer>(name / "add_noise_to_sigmoid_input",
                                                             ElementWiseLayerParams{ { name / "transposed_energy", name / "scaled_noise" }, { name / "noisy_energy" } });
        networkParameters.mWorkflow.add<SigmoidActivation>(name / "activate_energy_using_sigmoid", BasicParams{ { name / "noisy_energy" }, { name / "activated_energy" } });
        // Calculate alignments
        networkParameters.mWorkflow.add<ElementWiseMulLayer>(name / "calculate_first_part_of_alignments",
                                                             ElementWiseLayerParams{ { stateName, name / "activated_energy" }, { name / "first_part_of_alignments" } });
        // Reverse activated energy
        networkParameters.mWorkflow.add<lsa::LSAConstantsInitializerLayer>(name / "create_constant", BasicParams{ {}, { name / "ones" } });
        networkParameters.mWorkflow.add<ElementWiseSubLayer>(name / "reverse_activated_energy",
                                                             ElementWiseLayerParams{ { name / "ones", name / "activated_energy" }, { name / "reversed_activated_energy" } });
        networkParameters.mWorkflow.add<ElementWiseMulLayer>(name / "calculate_second_part_of_alignments",
                                                             ElementWiseLayerParams{ { stateName, name / "reversed_activated_energy" }, { name / "second_part_of_alignments" } });
        networkParameters.mWorkflow.add<RollLayer>(name / "shift_second_part_of_alignments",
                                                   RollLayerParams{ { name / "second_part_of_alignments" }, { name / "shifted_second_part_of_alignments" }, Dimension::Width, 1u, false, 0.0_dt });
        networkParameters.mWorkflow.add<ElementWiseSumLayer>(name / "calculate_alignments",
                                                             ElementWiseLayerParams{ { name / "first_part_of_alignments", name / "shifted_second_part_of_alignments" }, { alignmentsName } });
    }
    else
    {
        if (params.mSmoothing)
        {
            networkParameters.mWorkflow.add<SigmoidActivation>(name / "activate_energy_using_sigmoid", BasicParams{ { name / "transposed_energy" }, { name / "activated_energy" } });
            // Calculate divisor
            networkParameters.mWorkflow.add<ReduceSumLayer>(name / "calculate_divisor_for_energy", BasicParamsWithDim{ { name / "activated_energy" }, { name / "energy_divisor" }, Dimension::Width });
            // Normalize energy
            networkParameters.mWorkflow.add<ElementWiseDivLayer>(name / "smoothing_normalization",
                                                                 ElementWiseLayerParams{ { name / "activated_energy", name / "energy_divisor" }, { alignmentsName } });
        }
        else
        {
            networkParameters.mWorkflow.add<SoftMaxActivation>(name / "probability_fn", BasicParamsWithDim{ { name / "transposed_energy" }, { alignmentsName }, Dimension::Width });
        }
    }

    if (params.mUseForward)
    {
        // Shift previous alignments (i.e. input state)
        networkParameters.mWorkflow.add<RollLayer>(name / "shift_state", RollLayerParams{ { stateName }, { name / "previous_alignments_shifted" }, Dimension::Width, 1u, false, 0.0_dt });
        if (!params.mHparams.mUseStepwiseMonotonicConstraintType)
        {
            networkParameters.mWorkflow.add<lsa::LSAConstantsInitializerLayer>(name / "create_constant", BasicParams{ {}, { name / "ones" } });
        }
        networkParameters.mWorkflow.add<ElementWiseSubLayer>(name / "reverse_transit_proba", ElementWiseLayerParams{ { name / "ones", name / "transit_proba" }, { name / "reversed_transit_proba" } });
        networkParameters.mWorkflow.add<ElementWiseMulLayer>(name / "weight_previous_alignments", ElementWiseLayerParams{ { stateName, name / "reversed_transit_proba" }, { stateName / "weighted" } });
        networkParameters.mWorkflow.add<ElementWiseMulLayer>(
            name / "weight_previous_alignments_shifted",
            ElementWiseLayerParams{ { name / "previous_alignments_shifted", name / "transit_proba" }, { name / "previous_alignments_shifted_and_weighted" } });
        networkParameters.mWorkflow.add<ElementWiseSumLayer>(
            name / "sum_weighted_previous_alignments",
            ElementWiseLayerParams{ { stateName / "weighted", name / "previous_alignments_shifted_and_weighted" }, { name / "weighted_sum_of_previous_alignments" } });
        networkParameters.mWorkflow.add<ElementWiseMulLayer>(name / "scale_by_newly_calculated_alignments",
                                                             ElementWiseLayerParams{ { alignmentsName, name / "weighted_sum_of_previous_alignments" }, { name / "unnormalized_alignments" } });
        // Normalize calculated score
        networkParameters.mWorkflow.add<ReduceSumLayer>(name / "calculate_divisor_for_alignments",
                                                        BasicParamsWithDim{ { name / "unnormalized_alignments" }, { name / "alignments_divisor" }, Dimension::Width });
        // Normalize energy
        networkParameters.mWorkflow.add<ElementWiseDivLayer>(name / "calculate_final_alignments",
                                                             ElementWiseLayerParams{ { name / "unnormalized_alignments", name / "alignments_divisor" }, { params.getOutputs()[0] } });
    }

    if (mCumulativeMode)
    {
        const size_t i = mHasMask ? 2 : 1;

        // Get new state
        networkParameters.mWorkflow.add<raul::ElementWiseSumLayer>(name / "calculate_new_state", raul::ElementWiseLayerParams{ { params.getOutputs()[0], stateName }, { params.getOutputs()[i] } });
    }

    if (params.getOutputs().size() == 4 || ((!mHasMask || !mCumulativeMode) && params.getOutputs().size() == 3) || (!mHasMask && !mCumulativeMode && params.getOutputs().size() == 2))
    {
        networkParameters.mWorkflow.tensorNeeded(name, params.getOutputs().back(), WShape{ BS(), 1u, 1u, 1u }, DEC_FORW_WRIT);

        // Get max attention indices
        networkParameters.mWorkflow.add<ArgMaxLayer>(name / "get_max_attn", BasicParamsWithDim{ { params.getOutputs()[0] }, { params.getOutputs().back() }, Dimension::Width });
    }
}

}