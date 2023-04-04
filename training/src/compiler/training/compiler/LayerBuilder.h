// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef LAYER_BUILDER_H
#define LAYER_BUILDER_H

#include <functional>
#include <memory>

#include <training/system/Name.h>

namespace raul
{

class BasicLayer;
struct BasicParams;
struct NetworkParameters;

typedef std::unique_ptr<BasicLayer> LayerMem;

struct BasicLayerInfo
{
    BasicLayerInfo(const Names& in, const Names& out)
        : inputs(in)
        , outputs(out)
    {
    }

    Names inputs;
    Names outputs;
};

class BasicLayerBuilder
{
  public:
    BasicLayerBuilder(const Name& name, const BasicParams& params);
    ~BasicLayerBuilder(){}
    const Name& getName() const { return mName; }

    BasicLayerInfo& getParams() { return mParams; }
    const BasicLayerInfo& getParams() const { return mParams; }

    LayerMem build(NetworkParameters& networkParameters) const;
 
  protected:
    static void alterBasicParam(const BasicLayerInfo& param, BasicParams& bParams);

    Name mName;
    BasicLayerInfo mParams;
    std::function<LayerMem(NetworkParameters&, const BasicLayerInfo& param)> mConstr;
};

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4100)
#endif

template<typename T, class... Args>
class LayerBuilder : public BasicLayerBuilder
{
  public:
    LayerBuilder(const Name& name, Args&&... args)
        : BasicLayerBuilder(name, getParamFromArgs(args...))
    {
        mConstr = [name, args...](NetworkParameters& networkParameters, const BasicLayerInfo& param) mutable {
            BasicLayerBuilder::alterBasicParam(param, getParamFromArgs(args...));
            return std::make_unique<T>(name, std::forward<Args>(args)..., networkParameters);
        };
    }
    ~LayerBuilder(){}
  private:
    template<typename T0, typename... TT>
    static T0& getParamFromArgs(T0& t, [[maybe_unused]] TT&&... args)
    {
        return t;
    }
};
#ifdef _MSC_VER
#pragma warning(pop)
#endif

} // raul namespace

#endif
