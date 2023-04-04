// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_MODELOPTIMIZER
#define _H_MODELOPTIMIZER

#include <memory>
#include "model_common.h"
#include "OPOptimizers/DeprecatedOPOptimizer.hpp"
#include "OPOptimizers/WeightBNOptimizer.hpp"
#include "OPOptimizers/BNScaleOptimizer.hpp"
#include "OPOptimizers/WeightScaleOptimizer.hpp"
#include "OPOptimizers/PadOptimizer.hpp"
#include "OPOptimizers/InPlaceOptimizer.hpp"
#include "OPOptimizers/ActivationOptimizer.hpp"
#include "OPOptimizers/ChannelPaddingOptimizer.hpp"
#include "OPOptimizers/DepthwisePointwiseOptimizer.hpp"
#include "OPOptimizers/TransposeMulToScaleOptimizer.hpp"
#include "OPOptimizers/TransposeMatMulToFCOptimizer.hpp"
#include "OPOptimizers/FCFCOptimizer.hpp"
#include "OPOptimizers/ClipOptimizer.hpp"
#include "OPOptimizers/ReshapeOptimizer.hpp"
#include "OPOptimizers/QuantizationOptimizer.hpp"
#include "OPOptimizers/MemoryReuseOptimizer.hpp"
#include "OPOptimizers/ShGaUnCoReOptimizer.hpp"
#include "OPOptimizers/RNNOptimizer.hpp"
#include "OPOptimizers/LayerNormOptimizer.hpp"
#include "OPOptimizers/InnerProductOptimizer.hpp"
#include "OPOptimizers/GeluOptimizer.hpp"
#include "OPOptimizers/InvariantSliceOptimizer.hpp"
#include "OPOptimizers/MultiHeadAttentionOptimizer.hpp"
#include "OPOptimizers/StdDeviationOptimizer.hpp"
#include "OPOptimizers/PowerOptimizer.hpp"
#include "OPOptimizers/ConvolutionStrideOptimizer.hpp"
#include "OPOptimizers/MergeSameAndScaleOPOptimizer.hpp"
#include "OPOptimizers/ConvolutionEltwiseOptimizer.hpp"
#include "OPOptimizers/SpliceFCOptimizer.hpp"
#include "OPOptimizers/DilationConvOptimizer.hpp"
#include "OPOptimizers/ConvolutionSliceOptimizer.hpp"
#include "OPOptimizers/SignOptimizer.hpp"
#include "OPOptimizers/SwapPadTransposeOptimizer.hpp"
#include "OPOptimizers/ResizeFuseOptimizer.hpp"
#include "OPOptimizers/TransposeOptimizer.hpp"
#include "OPOptimizers/InputTransOptimizer.hpp"
#include "OPOptimizers/SwapTransposeEltOptimizer.hpp"
#include "OPOptimizers/SwapChannelResizePoolingOptimizer.hpp"
#include "OPOptimizers/TransConcatTransOptimizer.hpp"
#include "OPOptimizers/SwishOptimizer.hpp"
#include "OPOptimizers/HSwishOptimizer.hpp"
#include "OPOptimizers/HSigmoidOptimizer.hpp"
#include "OPOptimizers/ReorderChannelResizeOptimizer.hpp"
//#include "OPOptimizers/ScaleWeightOptimizer.hpp"
#include "OPOptimizers/ModifyDtOfInputOptimizer.hpp"
#include "OPOptimizers/ReshapeINOptimizer.hpp"
#include "OPOptimizers/FuseReshapeOptimizer.hpp"
#include "OPOptimizers/RsqrtOptimizer.hpp"
#include "OPOptimizers/MergeSharedWeightOptimizer.hpp"
#include "OPOptimizers/GATOptimizer.hpp"
#include "OPOptimizers/ConvConvOptimizer.hpp"
#include "OPOptimizers/ConvFCOptimizer.hpp"
#include "OPOptimizers/TransposeConvOptimizer.hpp"
#include "OPOptimizers/CleanInputsOutputsOptimizer.hpp"
#include "OPOptimizers/ConcatConvolutionOptimizer.hpp"
#include "OPOptimizers/EltwiseConstantOptimizer.hpp"
#include "OPOptimizers/ReshapeReduceMeanOptimizer.hpp"
#include "OPOptimizers/AdvancedLayerNormOptimizer.hpp"
#include "OPOptimizers/ConstantFuseOptimizer.hpp"
#include "OPOptimizers/WhereSoftmaxWhereOptimizer.hpp"
#include "OPOptimizers/Dynamic1ReshapeOptimizer.hpp"
#include "OPOptimizers/Dynamic2ReshapeOptimizer.hpp"

class ModelSpecOptimizer {
public:
    ModelSpecOptimizer()
    {}

    bool optimize(ModelSpec *spec)
    {
        std::string originalInputs = std::string(spec->input_names[0]);
        std::string originalOutputs = std::string(spec->output_names[0]);
        for (int i = 1; i < spec->num_inputs; i++) {
            originalInputs = originalInputs + "," + std::string(spec->input_names[i]);
        }
        for (int i = 1; i < spec->num_outputs; i++) {
            originalOutputs = originalOutputs + "," + std::string(spec->output_names[i]);
        }
        int originalInputSize = spec->num_inputs;
        int originalOutputSize = spec->num_outputs;

        // kernel code
        bool optimizeOrNot = false;
        for (auto opo : opos) {
            auto &ptr = *opo.get();
            const char *classNameAll = typeid(ptr).name();
            char *className;
            strtol(classNameAll, &className, 10);
            UNI_DEBUG_LOG("run optimizer: %s.\n", className);
            if (opo->optimize(spec)) {
                optimizeOrNot = true;
            }
        }

        if (originalInputSize == spec->num_inputs && originalOutputSize == spec->num_outputs) {
            modify_ms_inputs_and_outputs(spec, originalInputs, originalOutputs);
        }
        return optimizeOrNot;
    }

    void suggest(bool isPTQ)
    {
        // strict order
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new CleanInputsOutputsOptimizer()));
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new ResizeFuseOptimizer()));
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new ConstantFuseOptimizer()));
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new DeprecatedOPOptimizer()));

        this->opos.push_back(std::shared_ptr<OPOptimizer>(new ConcatConvolutionOptimizer()));
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new RNNOptimizer()));
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new GATOptimizer()));
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new RsqrtOptimizer()));
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new FuseReshapeOptimizer()));
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new ModifyDtOfInputOptimizer()));
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new ClipOptimizer()));
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new TransposeOptimizer()));
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new SwapTransposeEltOptimizer()));
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new SwapPadTransposeOptimizer()));
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new TransConcatTransOptimizer()));
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new TransposeOptimizer()));
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new DilationConvolutionOptimizer()));
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new SignOptimizer()));
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new GeluOptimizer()));
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new SwishOptimizer()));
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new HSwishOptimizer()));
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new HSigmoidOptimizer()));
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new TransposeMatMulToFCOptimizer()));
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new InnerProductOptimizer()));
        // this->opos.push_back(std::shared_ptr<OPOptimizer>(new MultiHeadAttentionOptimizer()));
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new InvariantSliceOptimizer()));
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new PowerOptimizer()));
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new SpliceFCOptimizer()));
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new AdvancedLayerNormOptimizer()));
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new TransposeOptimizer()));
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new ActivationOptimizer()));
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new ConvolutionSliceOptimizer()));
        if (!isPTQ) {
            // Fuse BN with previous conv or fc
            this->opos.push_back(std::shared_ptr<OPOptimizer>(new WeightBNOptimizer()));
            // Fuse scale with previous conv or fc
            this->opos.push_back(std::shared_ptr<OPOptimizer>(new WeightScaleOptimizer()));
            this->opos.push_back(std::shared_ptr<OPOptimizer>(new WeightBNOptimizer()));
            this->opos.push_back(std::shared_ptr<OPOptimizer>(new WeightScaleOptimizer()));
            this->opos.push_back(std::shared_ptr<OPOptimizer>(new ActivationOptimizer()));
            this->opos.push_back(std::shared_ptr<OPOptimizer>(new BNScaleOptimizer()));
        }
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new WeightScaleOptimizer(isPTQ)));
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new PadOptimizer()));

        this->opos.push_back(std::shared_ptr<OPOptimizer>(new ActivationOptimizer()));
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new ConvConvOptimizer()));
        if (!isPTQ) {
            this->opos.push_back(std::shared_ptr<OPOptimizer>(new ChannelPaddingOptimizer()));
            //this->opos.push_back(std::shared_ptr<OPOptimizer>(new ScaleWeightOptimizer()));
            this->opos.push_back(std::shared_ptr<OPOptimizer>(new DepthwisePointwiseOptimizer()));
        }
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new TransposeMulToScaleOptimizer()));

        this->opos.push_back(std::shared_ptr<OPOptimizer>(new LayerNormOptimizer()));
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new ReshapeOptimizer()));
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new ShGaUnCoReOptimizer()));
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new StdDeviationOptimizer()));

        this->opos.push_back(std::shared_ptr<OPOptimizer>(new MergeSameAndScaleOPOptimizer()));
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new ConvolutionStrideOptimizer()));
        // this->opos.push_back(std::shared_ptr<OPOptimizer>(new ReshapeINOptimizer()));
        //this->opos.push_back(std::shared_ptr<OPOptimizer>(new FCFCOptimizer()));
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new ConvFCOptimizer()));
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new TransposeConvOptimizer()));
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new MergeSharedWeightOptimizer()));
        // this->opos.push_back(std::shared_ptr<OPOptimizer>(new ConvolutionEltwiseOptimizer()));
        // this->opos.push_back(std::shared_ptr<OPOptimizer>(new ReorderChannelResizeOptimizer()));
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new SwapChannelResizePoolingOptimizer()));
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new InputTransOptimizer()));
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new EltwiseConstantOptimizer()));
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new ReshapeReduceMeanOptimizer()));
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new WhereSoftmaxWhereOptimizer()));
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new Dynamic1ReshapeOptimizer()));
	    this->opos.push_back(std::shared_ptr<OPOptimizer>(new Dynamic2ReshapeOptimizer()));

        // Please leave MemoryReuseOptimizer at last
        if (!isPTQ) {
            this->opos.push_back(std::shared_ptr<OPOptimizer>(new InPlaceOptimizer()));
            this->opos.push_back(std::shared_ptr<OPOptimizer>(new MemoryReuseOptimizer()));
        }
    }

    void suggest_for_training()
    {
        // strict order
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new DeprecatedOPOptimizer()));

        this->opos.push_back(std::shared_ptr<OPOptimizer>(new PadOptimizer()));

        this->opos.push_back(std::shared_ptr<OPOptimizer>(new MemoryReuseOptimizer()));
        // this->opos.push_back(std::shared_ptr<OPOptimizer>(new QuantizationOptimizer()));
    }

    void suggest_for_ptq(std::string inferPrecision, bool fuseBN, const char *scaleFile, F32 clipVal)
    {
        if (fuseBN) {
            // Fuse BN with previous conv or fc
            this->opos.push_back(std::shared_ptr<OPOptimizer>(new WeightBNOptimizer()));
            // Fuse scale with previous conv or fc
            this->opos.push_back(std::shared_ptr<OPOptimizer>(new WeightScaleOptimizer()));
            this->opos.push_back(std::shared_ptr<OPOptimizer>(new WeightBNOptimizer()));
            this->opos.push_back(std::shared_ptr<OPOptimizer>(new ActivationOptimizer()));
            this->opos.push_back(std::shared_ptr<OPOptimizer>(new ConvolutionStrideOptimizer()));
        }
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new ChannelPaddingOptimizer()));

        bool hiddenMode = (inferPrecision == "HIDDEN");
        //if (!hiddenMode) {
        //    this->opos.push_back(std::shared_ptr<OPOptimizer>(new DepthwisePointwiseOptimizer()));
        //}
        if ((inferPrecision == "INT8") || (inferPrecision == "INT4")) {
            this->opos.push_back(
                std::shared_ptr<OPOptimizer>(new QuantizationOptimizer(hiddenMode, scaleFile, clipVal)));
        }
        this->opos.push_back(std::shared_ptr<OPOptimizer>(new MemoryReuseOptimizer()));
    }

    void empty()
    {}

private:
    std::vector<std::shared_ptr<OPOptimizer>> opos;
};

#endif
