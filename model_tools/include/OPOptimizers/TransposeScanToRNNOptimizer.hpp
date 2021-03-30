// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_TRANSPOSESCANTORNNOPTIMIZER
#define _H_TRANSPOSESCANTORNNOPTIMIZER

#include "OPOptimizer.hpp"

/*  fuse transpose + onnx scan -> rnn

          |               
      transpose           
          |                     |
         scan         ---->    rnn
          |                     |
      transpose
          |          

    * do not support single onnx scan op now!
*/

class TransposeScanToRNNOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        for (int i = 1; i < spec->num_operator_specs; i++) {
            // do not support single onnx scan op
            if (spec->ops[i].type == OT_RNN) {
                // this is not a scan op
                if ((spec->ops[i].num_inputs < 2) || (spec->ops[i].num_outputs < 2)) {
                    continue;
                }
                std::string curIn = spec->ops[i].input_tensors_name[0];
                auto preOpIndexes = searchOperatorIndexByOutput(spec, curIn, 0, i, false);
                if ((preOpIndexes.size() != 1) || (-1 == preOpIndexes[0].first) ||
                    (spec->ops[preOpIndexes[0].first].type != OT_Transpose)) {
                    UNI_ERROR_LOG("There is a ONNX Scan op %s without two Tanspose ops, not "
                                  "support.",
                        spec->ops[i].name);
                }
            }
        }

        const int queryNum = 1;
        OperatorType queryOps[queryNum] = {OT_RNN};
        bool hasOptimized = false;
        for (int i = 1; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_Transpose) {
                int transposeOp0Index = i;
                std::string curOut = spec->ops[i].output_tensors_name[0];
                auto nextOpIndexes = searchOperatorIndexByInput(
                    spec, curOut, i + 1, spec->num_operator_specs, false);

                // there is not only a rnn op after this transpose
                if ((nextOpIndexes.size() != 1) || (-1 == nextOpIndexes[0].first) ||
                    (spec->ops[nextOpIndexes[0].first].type != OT_RNN)) {
                    continue;
                }

                int scanOpIndex = nextOpIndexes[0].first;

                // this is not a scan op
                if ((spec->ops[scanOpIndex].num_inputs < 2) ||
                    (spec->ops[scanOpIndex].num_outputs < 2)) {
                    continue;
                }

                curOut = spec->ops[scanOpIndex].output_tensors_name[0];
                nextOpIndexes = searchOperatorIndexByInput(
                    spec, curOut, scanOpIndex + 1, spec->num_operator_specs, false);

                // there is only one transpose op
                if ((nextOpIndexes.size() != 1) || (-1 == nextOpIndexes[0].first) ||
                    (spec->ops[nextOpIndexes[0].first].type != OT_Transpose)) {
                    UNI_ERROR_LOG("There is a ONNX Scan op without two Tanspose ops, not support.");
                }

                int transposeOp1Index = nextOpIndexes[0].first;
                hasOptimized = true;

                setOperatorInvalid(spec, transposeOp0Index, true);
                setOperatorInvalid(spec, transposeOp1Index, true);
            }
        }
        return hasOptimized;
    }
};
#endif
