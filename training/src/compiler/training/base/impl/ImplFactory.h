// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef IMPL_FACTORY_H
#define IMPL_FACTORY_H

#include <memory>
#include <typeinfo>
#include <unordered_map>

#include <training/system/Name.h>

#include <training/system/Singleton.h>
#include <training/system/Errors.h>

namespace raul
{
class BasicLayer;
class BasicImpl;

#define REGISTER_BUILDER(nameFront, nameImpl, map)                                                                                                                                                     \
    struct ImplBuilder##nameFront##map : public ImplBuilderBasic                                                                                                                                       \
    {                                                                                                                                                                                                  \
        std::unique_ptr<BasicImpl> create(BasicLayer* frontLayer) const override                                                                                                                       \
        {                                                                                                                                                                                              \
            nameFront* front = static_cast<nameFront*>(frontLayer);                                                                                                                                    \
            return std::make_unique<nameImpl>(*front);                                                                                                                                                 \
        }                                                                                                                                                                                              \
    };                                                                                                                                                                                                 \
    return map.insert({ typeid(nameFront).name(), std::make_unique<ImplBuilder##nameFront##map>() }).second;

class ImplFactory
{
  public:
    template<typename nameFront, typename nameImpl>
    bool regCPUFP32()
    {
        if (mMapImplCPUFP32.find(typeid(nameFront).name()) != mMapImplCPUFP32.end())
        {
            THROW_NONAME("ImplFactory", Name("Front ") + typeid(nameFront).name() + " already registered");
        }
        REGISTER_BUILDER(nameFront, nameImpl, mMapImplCPUFP32)
    }

    template<typename nameFront, typename nameImpl>
    bool regCPUFP16()
    {
        if (mMapImplCPUFP16.find(typeid(nameFront).name()) != mMapImplCPUFP16.end())
        {
            THROW_NONAME("ImplFactory", Name("Front ") + typeid(nameFront).name() + " already registered");
        }
        REGISTER_BUILDER(nameFront, nameImpl, mMapImplCPUFP16)
    }

    /**
     * @brief Input/output FP32, calculation FP16
     */
    template<typename nameFront, typename nameImpl>
    bool regCPUFP32FP16MixedLocal()
    {
        if (mMapImplCPUFP32FP16MixedLocal.find(typeid(nameFront).name()) != mMapImplCPUFP32FP16MixedLocal.end())
        {
            THROW_NONAME("ImplFactory", Name("Front ") + typeid(nameFront).name() + " already registered");
        }
        REGISTER_BUILDER(nameFront, nameImpl, mMapImplCPUFP32FP16MixedLocal)
    }

    void clearRegistrationFromEveryMap(const Name& name);

    struct ImplBuilderBasic
    {
        virtual std::unique_ptr<BasicImpl> create(BasicLayer* frontLayer) const = 0;
        virtual ~ImplBuilderBasic() {}
    };

    typedef std::unordered_map<Name, std::unique_ptr<ImplBuilderBasic>> MapImpl;

    MapImpl& getCPUFP32Map() { return mMapImplCPUFP32; }
    MapImpl& getCPUFP16Map() { return mMapImplCPUFP16; }
    MapImpl& getCPUFP32FP16MixedLocalMap() { return mMapImplCPUFP32FP16MixedLocal; }

  private:
    ImplFactory() {}
    ~ImplFactory() {}
    ImplFactory(const ImplFactory&) = delete;
    ImplFactory& operator=(const ImplFactory&) = delete;
    ImplFactory* operator&() = delete;

    MapImpl mMapImplCPUFP32;
    MapImpl mMapImplCPUFP16;
    MapImpl mMapImplCPUFP32FP16MixedLocal; // input/output FP32, calculation FP16

    friend struct CreateStatic<ImplFactory>;
};

typedef SingletonHolder<ImplFactory, CreateStatic, PhoenixSingleton> TheImplFactory;
} // raul namespace

#endif