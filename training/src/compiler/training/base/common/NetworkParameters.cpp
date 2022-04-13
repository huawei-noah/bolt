// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "NetworkParameters.h"

namespace raul
{

std::function<void(raul::BasicLayer*, raul::MemoryManager&, raul::NetworkParameters::CallbackPlace)>
CallbackHelper(std::optional<std::function<void(raul::BasicLayer*, raul::MemoryManager&)>> beforeForward,
               std::optional<std::function<void(raul::BasicLayer*, raul::MemoryManager&)>> afterForward,
               std::optional<std::function<void(raul::BasicLayer*, raul::MemoryManager&)>> beforeBackward,
               std::optional<std::function<void(raul::BasicLayer*, raul::MemoryManager&)>> afterBackward)
{
    return [=](raul::BasicLayer* layer, raul::MemoryManager& mem, raul::NetworkParameters::CallbackPlace place) {
        switch (place)
        {
            case raul::NetworkParameters::CallbackPlace::Before_Forward:
                if (beforeForward)
                {
                    beforeForward.value()(layer, mem);
                }
                break;
            case raul::NetworkParameters::CallbackPlace::After_Forward:
                if (afterForward)
                {
                    afterForward.value()(layer, mem);
                }
                break;
            case raul::NetworkParameters::CallbackPlace::Before_Backward:
                if (beforeBackward)
                {
                    beforeBackward.value()(layer, mem);
                }
                break;
            case raul::NetworkParameters::CallbackPlace::After_Backward:
                if (afterBackward)
                {
                    afterBackward.value()(layer, mem);
                    ;
                }
                break;
            default:
                THROW_NONAME("NetworkParameters", "unknown place to use callback");
        }
    };
}

}