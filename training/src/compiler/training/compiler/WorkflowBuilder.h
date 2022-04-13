// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef WORKFLOW_BUILDER_H
#define WORKFLOW_BUILDER_H

#include <training/frontend/Frontend.h>
#include <training/compiler/LayerBuilder.h>

#include <training/base/layers/BasicLayer.h>
#include <training/base/layers/activations/SigmoidActivation.h>
#include <training/base/layers/activations/SoftMaxActivation.h>
#include <training/base/layers/activations/TanhActivation.h>
#include <training/base/layers/basic/ReshapeLayer.h>
#include <training/base/layers/basic/trainable/LinearLayer.h>
#include <training/compiler/Workflow.h>

#include <training/system/Errors.h>

namespace raul
{

struct WorkflowBuilder : frontend::Processor
{
    explicit WorkflowBuilder(Workflow& workflow)
        : work{ workflow }
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
                THROW_NONAME("WorkflowBuilder", ss.str());
            }
        }
    }

    void process(const frontend::LinearDeclaration& x, const std::optional<frontend::Path> path) override
    {
        auto name = getName(path);
        const auto inputNames = getSourcePorts(path, x.inputs);
        work.add<LinearLayer>(name, LinearParams{ inputNames[0], name / x.outputs[0], x.features, x.bias });
    }

    void process(const frontend::SigmoidDeclaration& x, const std::optional<frontend::Path> path) override
    {
        const auto name = getName(path);
        const auto inputNames = getSourcePorts(path, x.inputs);
        work.add<SigmoidActivation>(name, BasicParams{ { inputNames[0] }, Names{ name / x.outputs[0] } });
    }

    void process(const frontend::TanhDeclaration& x, const std::optional<frontend::Path> path) override
    {
        const auto name = getName(path);
        const auto inputNames = getSourcePorts(path, x.inputs);
        work.add<TanhActivation>(name, BasicParams{ { inputNames[0] }, Names{ name / x.outputs[0] } });
    }

    void process(const frontend::SoftmaxDeclaration& x, const std::optional<frontend::Path> path) override
    {
        const auto name = getName(path);
        const auto inputNames = getSourcePorts(path, x.inputs);
        work.add<SoftMaxActivation>(name, BasicParamsWithDim{ { inputNames[0] }, Names{ name / x.outputs[0] } });
    }

    void process(const frontend::ReshapeDeclaration& x, const std::optional<frontend::Path> path) override
    {
        const auto name = getName(path);
        const auto inputNames = getSourcePorts(path, x.inputs);
        if (x.shape.size() != 3)
        {
            THROW_NONAME("WorkflowBuilder", "workflow support 3 component tensor reshape");
        }
        work.add<ReshapeLayer>(name, ViewParams{ inputNames[0], name / x.outputs[0], x.shape[0], x.shape[1], x.shape[2] });
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
                THROW_NONAME("WorkflowBuilder", ss.str());
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
                THROW_NONAME("WorkflowBuilder", ss.str());
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

  private:
    std::map<Name, Name> currentPorts;
    Workflow& work;
};

} // namespace raul

#endif // WORKFLOW_BUILDER_H