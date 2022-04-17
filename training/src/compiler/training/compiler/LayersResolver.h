// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef WORKFLOW_LAYERS_RESOLVER_H
#define WORKFLOW_LAYERS_RESOLVER_H

#include <training/compiler/LayerBuilder.h>
#include <training/frontend/Frontend.h>

#include <training/base/layers/BasicLayer.h>
#include <training/base/layers/activations/SigmoidActivation.h>
#include <training/base/layers/activations/SoftMaxActivation.h>
#include <training/base/layers/activations/TanhActivation.h>
#include <training/base/layers/basic/DataLayer.h>
#include <training/base/layers/basic/ReshapeLayer.h>
#include <training/base/layers/basic/trainable/LinearLayer.h>
#include <training/compiler/Workflow.h>

#include <training/system/Errors.h>

namespace raul
{

struct LayersResolver : frontend::Processor
{
    explicit LayersResolver(std::vector<BasicLayerBuilder>& init)
        : layers{ init }
    {
    }

    void process(const frontend::GraphDeclaration& graph, const std::optional<frontend::Path> path) override
    {
        prepareCurrentPorts(graph, path);

        for (auto& [elementName, elementGenerator] : graph.elements)
        {
            auto fullName = path ? *path / elementName : elementName;
            try
            {
                elementGenerator.apply(*this, fullName);
            }
            catch (...)
            {
                std::stringstream ss;
                ss << "cannot process element \"" << fullName << "\"";
                THROW_NONAME("LayersResolver", ss.str());
            }
        }
    }

    void process(const frontend::LinearDeclaration& x, const std::optional<frontend::Path> path) override
    {
        auto name = getName(path);
        const auto inputNames = getSourcePorts(path, x.inputs);
        layers.emplace_back(raul::LayerBuilder<LinearLayer, raul::LinearParams>(name, LinearParams{ inputNames[0], name / x.outputs[0], x.features, x.bias }));
    }

    void process(const frontend::SigmoidDeclaration& x, const std::optional<frontend::Path> path) override
    {
        const auto name = getName(path);
        const auto inputNames = getSourcePorts(path, x.inputs);
        layers.emplace_back(raul::LayerBuilder<SigmoidActivation, raul::BasicParams>(name, BasicParams{ { inputNames[0] }, Names{ name / x.outputs[0] } }));
    }

    void process(const frontend::TanhDeclaration& x, const std::optional<frontend::Path> path) override
    {
        const auto name = getName(path);
        const auto inputNames = getSourcePorts(path, x.inputs);
        layers.emplace_back(raul::LayerBuilder<TanhActivation, raul::BasicParams>(name, BasicParams{ { inputNames[0] }, Names{ name / x.outputs[0] } }));
    }

    void process(const frontend::SoftmaxDeclaration& x, const std::optional<frontend::Path> path) override
    {
        const auto name = getName(path);
        const auto inputNames = getSourcePorts(path, x.inputs);
        layers.emplace_back(raul::LayerBuilder<SoftMaxActivation, raul::BasicParamsWithDim>(name, BasicParamsWithDim{ { inputNames[0] }, Names{ name / x.outputs[0] } }));
    }

    void process(const frontend::ReshapeDeclaration& x, const std::optional<frontend::Path> path) override
    {
        const auto name = getName(path);
        const auto inputNames = getSourcePorts(path, x.inputs);
        if (x.shape.size() != 3)
        {
            THROW_NONAME("LayersResolver", "workflow support 3 component tensor reshape");
        }
        layers.emplace_back(raul::LayerBuilder<ReshapeLayer, raul::ViewParams>(name, ViewParams{ inputNames[0], name / x.outputs[0], x.shape[0], x.shape[1], x.shape[2] }));
    }

    //  private:
    static Name getName(const std::optional<frontend::Path>& path) { return path ? path->fullname("::") : "noname"; }

    void prepareCurrentPorts(const frontend::GraphDeclaration& graph, const std::optional<frontend::Path>& path)
    {
        for (const auto& [from, to] : graph.connections)
        {
            const auto toPath = path ? *path / to.getPath() : to.getPath();
            const auto fromPath = path ? *path / from.getPath() : from.getPath();
            const auto toName = toPath.fullname("::");
            auto fromName = fromPath.fullname("::");

            if (currentPorts.find(toName) != currentPorts.end())
            {
                std::stringstream ss;
                ss << "input tensor collision: ? -> \"" << toName << "\"";
                THROW_NONAME("LayersResolver", ss.str());
            }

            while (currentPorts.find(fromName) != currentPorts.end())
            {
                fromName = currentPorts[fromName];
            }

            if (toName == fromName)
            {
                std::stringstream ss;
                ss << "loop found: \"" << toName << "\""
                   << " -> \"" << toName << "\"";
                THROW_NONAME("LayersResolver", ss.str());
            }

            currentPorts[toName] = fromName;
        }
    }

    [[nodiscard]] frontend::PortNames getSourcePorts(const std::optional<frontend::Path>& path, const frontend::PortNames& ports)
    {
        frontend::PortNames results;
        for (const auto& port : ports)
        {

            std::string originalName = path ? (*path / port).fullname("::") : port;

            while (currentPorts.find(originalName) != currentPorts.end())
            {
                originalName = currentPorts[originalName];
            }
            results.emplace_back(originalName);
        }
        return results;
    }

    void resolveInputs(const std::initializer_list<size_t> shape = { 1, 1, 1, 1 })
    {
        std::unordered_map<Name, std::initializer_list<size_t>> init;

        if (layers.empty())
        {
            return;
        }

        for (const auto& input : layers[0].getParams().inputs)
        {
            init[input] = shape;
        }
        resolveInputs(init);
    }

    void resolveInputs(std::unordered_map<Name, std::initializer_list<size_t>> shapeDict)
    {
        std::optional<size_t> batchSize;
        if (layers.empty())
        {
            return;
        }
        for (const auto& input : layers[0].getParams().inputs)
        {
            const auto shape = shapeDict[input];
            if (std::distance(shape.begin(), shape.end()) != 4)
            {
                THROW_NONAME("LayerResolver", "core supports only 4 dimensional tensors")
            }

            if (!batchSize)
            {
                batchSize = *shape.begin();
            }

            layers.emplace(layers.begin(), raul::LayerBuilder<DataLayer, raul::DataParams>(input, DataParams{ { input }, *(shape.begin() + 1), *(shape.begin() + 2), *(shape.begin() + 3) }));
        }
    }

  private:
    std::map<Name, Name> currentPorts;
    std::vector<BasicLayerBuilder>& layers;
};

} // namespace raul

#endif // WORKFLOW_LAYERS_RESOLVER_H