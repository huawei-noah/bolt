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

#include <vector>
#include <memory>
#include "model_tools.h"
#include "model_serialize_deserialize.hpp"
#include "OPOptimizers/OPOptimizer.hpp"
#include "OPOptimizers/DeprecatedOPOptimizer.hpp"
#include "OPOptimizers/ConvBNOptimizer.hpp"
#include "OPOptimizers/BNScaleOptimizer.hpp"
#include "OPOptimizers/ConvScaleOptimizer.hpp"
#include "OPOptimizers/InPlaceOptimizer.hpp"
#include "OPOptimizers/ConvActivationOptimizer.hpp"
#include "OPOptimizers/ChannelPaddingOptimizer.hpp"
#include "OPOptimizers/DepthwisePointwiseOptimizer.hpp"
#include "OPOptimizers/TransposeMulToScaleOptimizer.hpp"
#include "OPOptimizers/TransposeMatMulToFCOptimizer.hpp"
#include "OPOptimizers/FlattenGemmOptimizer.hpp"
#include "OPOptimizers/ClipClipOptimizer.hpp"
#include "OPOptimizers/MemoryReuseOptimizer.hpp"


class ConvEltwisePoolingOptimizer: public OPOptimizer {
    virtual bool optimize(ModelSpec* spec) override {
        if(spec == nullptr)
            return false;

        bool hasOptimized = false;
        // TODO: add fusion(low priority)
        return hasOptimized;
    }
};


class FCEltwiseOptimizer: public OPOptimizer {
    virtual bool optimize(ModelSpec* spec) override {
        if(spec == nullptr)
            return false;

        bool hasOptimized = false;
        // TODO: add fusion(middle priority)
        return hasOptimized;
    }
};

class ModelSpecOptimizer {
    public:
        ModelSpecOptimizer() { }
        /**
         * @param model
         */
        bool optimize(ModelSpec* spec) {
            bool optimizeOrNot = false;
            for (auto opo: opos) {
                if (opo->optimize(spec)) {
                    optimizeOrNot = true;
                }
            }
            return optimizeOrNot;
        }

        void suggest() {
            // strict order
            this->opos.push_back(std::shared_ptr<OPOptimizer>(new DeprecatedOPOptimizer()));

            // Removing ConvBN leads to error
            this->opos.push_back(std::shared_ptr<OPOptimizer>(new ConvBNOptimizer()));
            this->opos.push_back(std::shared_ptr<OPOptimizer>(new BNScaleOptimizer()));
            this->opos.push_back(std::shared_ptr<OPOptimizer>(new ConvScaleOptimizer()));

            this->opos.push_back(std::shared_ptr<OPOptimizer>(new InPlaceOptimizer()));
            
            this->opos.push_back(std::shared_ptr<OPOptimizer>(new ConvActivationOptimizer()));
            this->opos.push_back(std::shared_ptr<OPOptimizer>(new ChannelPaddingOptimizer()));
            this->opos.push_back(std::shared_ptr<OPOptimizer>(new DepthwisePointwiseOptimizer()));
            this->opos.push_back(std::shared_ptr<OPOptimizer>(new TransposeMulToScaleOptimizer()));
            this->opos.push_back(std::shared_ptr<OPOptimizer>(new TransposeMatMulToFCOptimizer()));
            this->opos.push_back(std::shared_ptr<OPOptimizer>(new FlattenGemmOptimizer()));
            this->opos.push_back(std::shared_ptr<OPOptimizer>(new ClipClipOptimizer()));

            // Please leave MemoryReuseOptimizer at last
            this->opos.push_back(std::shared_ptr<OPOptimizer>(new MemoryReuseOptimizer()));
        }

        void empty() {}

    private:
        // ModelSpecOptimizer() { }
        /**
         * @param opo
         */
        std::vector<std::shared_ptr<OPOptimizer>> opos;
};

#endif
