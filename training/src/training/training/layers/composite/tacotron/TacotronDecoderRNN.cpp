// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "TacotronDecoderRNN.h"

#include <training/api/API.h>
#include <training/layers/basic/ElementWiseSumLayer.h>
#include <training/layers/basic/SplitterLayer.h>
#include <training/layers/composite/rnn/LSTMCellLayer.h>

namespace raul::tacotron
{
using namespace std;

void AddDecoderRNN(Workflow* work, const Name& name, const BasicParams& params, const Names& next_state, const TacotronParams& tparams)
{
    if (params.getInputs().size() != 1 + 2 * tparams.decoderLstmUnits.size())
    {
        THROW("Tacotron", name, "bad inputs count");
    }
    if (params.getOutputs().size() != 1)
    {
        THROW("Tacotron", name, "bad outputs count");
    }
    auto input = params.getInputs().front();
    // state contains 2 * params.decoderLayers tensors ([hidded, cell] for each layer)
    auto state = Names(params.getInputs().begin() + 1, params.getInputs().end());

    size_t N = tparams.decoderLstmUnits.size();

    auto parent = params.getSharedLayer().empty() ? Name() : params.getSharedLayer() / "zoneout_lstm_cell" / "0";

    LSTMCellParams lstmParams{ { { input, state[0], state[1] }, { next_state[0], next_state[1] }, parent, tparams.getLayerExecutionTarget() }, true, tparams.zoneoutRate, true, 1.f, tparams.frozen };

    for (size_t i = 0; i < N - 1; ++i)
    {
        lstmParams.getOutputs()[0] = next_state[2 * i];
        LSTMCellLayer(name / "zoneout_lstm_cell" / to_string(i), lstmParams, work->getNetworkParameters());
        auto hName = lstmParams.getOutputs()[0];

        string output = next_state[2 * i];
        if (tparams.useResidualRnn && i > 0)
        {
            output = name / "output" / to_string(i);
            work->add<ElementWiseSumLayer>(name / "residual" / to_string(i), ElementWiseLayerParams{ { lstmParams.getInputs()[0], next_state[2 * i] }, { output } });
        }
        // work->add<SplitterLayer>(name / "hidden_splitter" / to_string(i), BasicParams{ { hName }, { hName / "1", next_state[2 * i] } });

        parent = params.getSharedLayer().empty() ? Name() : params.getSharedLayer() / "zoneout_lstm_cell" / to_string(i + 1);

        lstmParams = LSTMCellParams{ { { output, state[2 * (i + 1)], state[2 * (i + 1) + 1] }, { next_state[2 * (i + 1)], next_state[2 * (i + 1) + 1] }, parent, tparams.getLayerExecutionTarget() },
                                     true,
                                     tparams.zoneoutRate,
                                     true,
                                     1_dt,
                                     tparams.frozen };
    }

    lstmParams.getOutputs() = { next_state[2 * (N - 1)], next_state[2 * (N - 1) + 1] };

    LSTMCellLayer(name / "zoneout_lstm_cell" / to_string(N - 1), lstmParams, work->getNetworkParameters());
    if (tparams.useResidualRnn && N > 1)
    {
        work->add<ElementWiseSumLayer>(name / "residual" / to_string(N - 1), ElementWiseLayerParams{ { lstmParams.getInputs()[0], next_state[2 * (N - 1)] }, { params.getOutputs()[0] } });
    }
    else
    {
        work->add<SplitterLayer>(name / "output_splitter", BasicParams{ { next_state[2 * (N - 1)] }, { params.getOutputs()[0] } });
    }
}

} // namespace raul::tacotron
