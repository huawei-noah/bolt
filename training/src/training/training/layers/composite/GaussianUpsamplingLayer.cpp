// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "GaussianUpsamplingLayer.h"
#include "tacotron/GaussianUpsamplingDistributionLayer.h"
#include <training/layers/basic/ConcatenationLayer.h>
#include <training/layers/basic/CumSumLayer.h>
#include <training/layers/basic/ElementWiseDivLayer.h>
#include <training/layers/basic/ElementWiseSumLayer.h>
#include <training/layers/basic/FixedBiasLayer.h>
#include <training/layers/basic/MatMulLayer.h>
#include <training/layers/basic/ReduceSumLayer.h>
#include <training/layers/basic/ScaleLayer.h>
#include <training/layers/basic/SlicerLayer.h>
#include <training/layers/basic/TileLayer.h>

#include <training/opencl/GPUCommon.h>

#include <cmath>

namespace raul
{

class GaussianUpsamplingConstantsInitializerLayer : public BasicLayer
{
  public:
    GaussianUpsamplingConstantsInitializerLayer(const Name& name, const BasicParams& params, const size_t melLen, NetworkParameters& networkParameters)
        : BasicLayer(name, "GaussianUpsamplingConstantsInitializer", params, networkParameters, { false, false })
    {
        auto prefix = "GaussianUpsamplingConstantsInitializerLayer[" + mName + "::ctor]: ";

        if (mOutputs.size() != 2)
        {
            THROW("GaussianUpsampling", name, "wrong number of output names");
        }
        if (mOutputs[0].empty() || mOutputs[1].empty())
        {
            THROW("GaussianUpsampling", name, "empty output name");
        }

        mLayerTarget = networkParameters.mWorkflow.getOverrideLayerExecutionTarget();

        // Declare needed constant
        networkParameters.mWorkflow.tensorNeeded(name, mOutputs[0], WShape{ BS(), 1u, 1u, 1u }, DEC_FORW_WRIT);
        networkParameters.mWorkflow.tensorNeeded(name, mOutputs[1], WShape{ melLen, 1u, 1u, 1u }, DEC_FORW_WRIT);
    }

    void forwardComputeImpl(NetworkMode) override
    {
        auto& work = mNetworkParams.mWorkflow;

        if (work.getExecutionTarget() == ExecutionTarget::CPU || mLayerTarget == LayerExecutionTarget::CPU)
        {
            mNetworkParams.mMemoryManager[mOutputs[0]] = 0_dt;
            Tensor& t = mNetworkParams.mMemoryManager[mOutputs[1]];
            for (size_t i = 0; i < t.size(); ++i)
            {
                t[i] = static_cast<dtype>(i + 1);
            }
        }
        else if (work.getExecutionTarget() == ExecutionTarget::GPU)
        {
            mNetworkParams.mMemoryManagerGPU[mOutputs[0]] = 0_dt;
            auto& t = work.getMemoryManager<MemoryManagerGPU>()(mOutputs[1]);
            gpu::iota(work.getKernelManager(), mTypeName + "[" + mName + "::forwardComputeImpl]", 1.0_dt, t.getBatchSize(), t.getBuffer());
        }
        else if (work.getExecutionTarget() == ExecutionTarget::CPUFP16 || mLayerTarget == LayerExecutionTarget::CPUFP16)
        {
            mNetworkParams.mMemoryManagerFP16[mOutputs[0]] = 0_hf;
            auto& t = mNetworkParams.mMemoryManagerFP16[mOutputs[1]];
            for (size_t i = 0; i < t.size(); ++i)
            {
                t[i] = static_cast<half>(i + 1);
            }
        }
        else
        {
            THROW("GaussianUpsampling", mName, "unsupported execution target");
        }
    }

    void backwardComputeImpl() override {}

    LayerExecutionTarget mLayerTarget;
};

GaussianUpsamplingLayer::GaussianUpsamplingLayer(const Name& name, const GaussianUpsamplingParams& params, NetworkParameters& networkParameters)
{
    auto prefix = "GaussianUpsampling[" + name + "::ctor]: ";

    // Input, Durations, Ranges, [MelLen]
    if (params.getInputs().size() != 3)
    {
        THROW("GaussianUpsampling", name, "wrong number of input names");
    }

    if (params.getOutputs().size() != 1)
    {
        THROW("GaussianUpsampling", name, "wrong number of output names");
    }

    // Input names
    auto [inputName, durationsName, rangesName] = std::make_tuple(params.getInputs()[0], params.getInputs()[1], params.getInputs()[2]);

    networkParameters.mWorkflow.add<ScaleLayer>(name / "divise_durations", ScaleParams{ { durationsName }, { name / "durations_divided" }, 0.5_dt });
    networkParameters.mWorkflow.add<CumSumLayer>(name / "csum_durations", BasicParamsWithDim{ { durationsName }, { name / "durations_cumulated" }, Dimension::Height });
    networkParameters.mWorkflow.add<SlicerLayer>(name / "slice_durations_cumulated",
                                                 SlicingParams{ { name / "durations_cumulated" },
                                                                { name / "durations_sliced", name / "durations_redundant" },
                                                                Dimension::Height,
                                                                { static_cast<int>(networkParameters.mWorkflow.getHeight(name / "durations_cumulated")) - 1, 1 } });
    networkParameters.mWorkflow.add<GaussianUpsamplingConstantsInitializerLayer>(name / "create_constants", BasicParams{ {}, { name / "zeroes", name / "t" } }, params.mMelLen);
    networkParameters.mWorkflow.add<ConcatenationLayer>(name / "get_durations_shifted",
                                                        BasicParamsWithDim{ { name / "zeroes", name / "durations_sliced" }, { name / "durations_shifted" }, Dimension::Height });
    networkParameters.mWorkflow.add<ElementWiseSumLayer>(name / "calculate_new_durations", ElementWiseLayerParams{ { name / "durations_divided", name / "durations_shifted" }, { name / "c" } });
    networkParameters.mWorkflow.add<FixedBiasLayer>(name / "get_ranges_biased", FixedBiasParams{ { rangesName }, { name / "ranges_biased" }, params.mEps });
    networkParameters.mWorkflow.add<GaussianUpsamplingDistributionLayer>(name / "calculate_distribution", BasicParams{ { name / "t", name / "c", name / "ranges_biased" }, { name / "p" } });
    networkParameters.mWorkflow.add<ReduceSumLayer>(name / "reduce_p", BasicParamsWithDim{ { name / "p" }, { name / "s_proto" }, Dimension::Width });
    networkParameters.mWorkflow.add<FixedBiasLayer>(name / "calculate_s", FixedBiasParams{ { name / "s_proto" }, { name / "s" }, params.mEps });

    auto name_s = name / "s";
    if (networkParameters.mWorkflow.getExecutionTarget() == ExecutionTarget::GPU && networkParameters.mWorkflow.getWidth(name / "p") != networkParameters.mWorkflow.getWidth(name_s))
    {
        networkParameters.mWorkflow.add<TileLayer>(name / "tile_s", TilingParams{ name_s, name_s + "tiled", networkParameters.mWorkflow.getWidth(name / "p"), Dimension::Width });
        name_s = name_s + "tiled";
    }

    networkParameters.mWorkflow.add<ElementWiseDivLayer>(name / "calculate_weights", ElementWiseLayerParams{ { name / "p", name_s }, { name / "w" } });
    networkParameters.mWorkflow.add<MatMulLayer>(name / "calculate_output", MatMulParams{ { name / "w", inputName }, { params.getOutputs()[0] } });
}

}
