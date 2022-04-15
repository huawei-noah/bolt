// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "JSON.h"

#include <training/frontend/Declaration.h>
#include <training/frontend/Generator.h>
#include <training/frontend/Graph.h>
#include <training/frontend/Layers.h>

namespace raul::frontend::io
{

template<class T>
std::optional<T> getOptional(const json& object, const std::string& key)
{
    if (object.find(key) != object.end())
    {
        return object[key];
    }

    return std::nullopt;
}

Generator LinearFromJSON(const json& object)
{
    size_t features = object["features"];

    auto layer = Linear{ features };

    auto bias = getOptional<bool>(object, "bias");

    if (bias && *bias)
    {
        layer = layer.enableBias();
    }

    return std::move(layer);
}

Generator GraphFromJSON(const json& object)
{
    auto graph = Graph{};

    for (const auto& [name, node] : object["nodes"].items())
    {
        graph[name] = fromJSON(node);
    }

    auto edges = getOptional<json>(object, "edges");
    if (edges)
    {
        auto createPort = [&](const json& x) -> Port
        {
            auto layer = getOptional<std::string>(x, "layer");
            auto port = static_cast<std::string>(x["port"]);

            if (layer)
            {
                return Port(*layer, port); // NOLINT(modernize-return-braced-init-list)
            }
            else
            {
                return Port(port);
            }
        };

        for (const auto& connection : *edges)
        {
            auto src = createPort(connection["from"]);
            auto dst = createPort(connection["to"]);
            auto connect = src.to(dst);
            graph.connect(connect);
        }
    }

    return std::move(graph);
}

std::optional<Type> getJSONObjectType(const json& object)
{
    auto type = getOptional<std::string>(object, "type");

    if (type)
    {
        if (*type == "linear") return Type::Linear;
        if (*type == "relu") return Type::ReLU;
        if (*type == "graph") return Type::Graph;
    }

    return std::nullopt;
}

Generator fromJSON(const json& object)
{
    auto elementType = getJSONObjectType(object);

    if (elementType)
    {
        if (*elementType == Type::Graph) return GraphFromJSON(object);
        if (*elementType == Type::Linear) return LinearFromJSON(object);
    }
    return Graph{};
}

} // namespace raul::frontend::io
