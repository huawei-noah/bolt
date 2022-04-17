// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef FRONTEND_GRAPH_H
#define FRONTEND_GRAPH_H

#include "Declaration.h"
#include "Generator.h"
#include "Port.h"
#include "Types.h"

#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <training/system/Errors.h>
#include <training/system/ordered_map.h>

namespace raul::frontend
{

/**
 * Graph element
 *
 * It explains a structure of a graph. An element can be a layer or
 * a subgraph. Element can have name (optional parameter).
 */
struct GraphElement
{
    /**
     * Ctor
     *
     * @param name element name
     * @param generator element (layer or graph)
     */
    GraphElement(Name name, Generator generator)
        : name{ std::move(name) }
        , generator{ std::move(generator) }
    {
    }

    /**
     * Ctor
     *
     * @param generator generator element (layer or graph)
     *
     * @note name is implicitly set (default value: "element" + id)
     */
    GraphElement(Generator generator)
        : name{ std::nullopt }
        , generator{ std::move(generator) }
    {
    }

    std::optional<Name> name;
    Generator generator;
};

/**
 * Graph declaration
 *
 * The declaration contains a map from the name of element to the name of subgraph.
 */
struct GraphDeclaration : Declaration
{
    using InterconnectType = std::pair<std::optional<Path>, PortNames>;

    /**
     * Ctor
     * @param init list of graph elements
     *
     * A graph element has two constructors (generator with default name, generator with specified name).
     */
    GraphDeclaration(std::initializer_list<GraphElement> init)
        : GraphDeclaration(init, {})
    {
        generateConnections();
    }

    GraphDeclaration(std::initializer_list<GraphElement> initElements, std::initializer_list<PortMap> initConnections)
        : Declaration(Type::Graph)
        , connections{ initConnections }
    {
        for (auto&& [k, el] : initElements)
        {
            auto name = k ? *k : genNextName();
            if (elements.contains(name))
            {
                std::stringstream ss;
                ss << "element names collision detected: \"" << name << "\" is already in the topology";
                THROW_NONAME("GraphDeclaration", ss.str());
            }
            elements[name] = el;
        }

        fillPorts();
    }

    [[nodiscard]] std::string genNextName() const
    {
        std::stringstream ss;
        ss << elements.size();
        return ss.str();
    }

  private:
    void connectLayers(const InterconnectType& from, const InterconnectType& to)
    {
        const auto& [fromName, fromPortNames] = from;
        const auto& [toName, toPortNames] = to;
        const auto size = std::min(fromPortNames.size(), toPortNames.size());
        for (size_t i = 0; i < size; ++i)
        {
            const auto& fromPortName = fromPortNames[i];
            const auto& toPortName = toPortNames[i];
            const auto fromPort = fromName ? Port(*fromName, fromPortName) : Port(fromPortName);
            const auto toPort = toName ? Port(*toName, toPortName) : Port(toPortName);
            connections.emplace_back(fromPort.to(toPort));
        }
    }

    void fillPorts()
    {
        std::unordered_set<Name> inputsMap;
        std::unordered_set<Name> outputsMap;

        for (const auto& [from, to] : connections)
        {
            if (!from.getLayer())
            {
                inputsMap.insert(from.getPort());
            }

            if (!to.getLayer())
            {
                outputsMap.insert(to.getPort());
            }
        }

        inputs.clear();
        outputs.clear();

        inputs.resize(inputsMap.size());
        outputs.resize(outputsMap.size());

        std::copy(inputsMap.cbegin(), inputsMap.cend(), inputs.begin());
        std::copy(outputsMap.cbegin(), outputsMap.cend(), outputs.begin());
    }

  public:
    void generateConnections()
    {
        connections.clear();

        std::optional<InterconnectType> interconnect = std::nullopt;

        for (auto&& [name, element] : elements)
        {
            const auto& elementInputs = element.getInputs();
            const auto& elementOutputs = element.getOutputs();

            auto from = interconnect ? *interconnect : InterconnectType{ std::nullopt, elementInputs };
            connectLayers(from, InterconnectType{ name, elementInputs });

            interconnect = InterconnectType{ name, PortNames(elementOutputs) };
        }

        if (interconnect)
        {
            connectLayers(*interconnect, InterconnectType{ std::nullopt, interconnect->second });
        }

        fillPorts();
    }

    system::ordered_map<Name, Generator> elements;
    std::vector<PortMap> connections;
};

/**
 * Graph generator
 */
struct Graph : GeneratorTyped<GraphDeclaration>
{
    /**
     * Ctor
     * @param init list of graph elements
     *
     * A graph element has two constructors (generator with default name, generator with specified name).
     */
    Graph(std::initializer_list<GraphElement> init)
        : GeneratorTyped(init)
    {
    }

    Graph(std::initializer_list<GraphElement> initElements, std::initializer_list<PortMap> initConnections)
        : GeneratorTyped(initElements, initConnections)
    {
    }

    /**
     * Unchecked access: operator[]
     * Elements in a Graph object can be accessed via operator[] similar to a std::vector.
     *
     * @param pos index of the graph element
     * @return reference to the graph element
     */
    Generator& operator[](size_t pos)
    {
        auto it = getDeclaration()->elements.begin();
        std::advance(it, pos);
        return it->second;
    }

    /**
     * Checked access: at()
     * Elements in a Graph object can be accessed via at() similar to a std::vector.
     *
     * @param pos index of the graph element
     * @return reference to the graph element
     */
    Generator& at(size_t pos)
    {
        const auto& elements = getDeclaration()->elements;
        if (pos >= elements.size())
        {
            std::stringstream ss;
            ss << "out of range: " << pos << " > " << elements.size();
            throw std::out_of_range(ss.str());
        }
        return operator[](pos);
    }

    /**
     * Access: operator[]
     * Elements in a Graph object can be accessed via operator[] similar to a std::map.
     *
     * @param name name of the elements
     * @return reference to the graph element
     */
    Generator& operator[](const Name& name) { return getDeclaration()->elements[name]; }

    // Generator& operator[](const Path& name) { return getDeclaration()->elements[name.str()]; }

    auto insert(const GraphElement& value)
    {

        auto name = value.name ? *value.name : getDeclaration()->genNextName();
        getDeclaration()->elements[name] = value.generator;
        getDeclaration()->generateConnections();
        return getDeclaration()->elements[name];
    }

    Graph connect(const PortMap& value)
    {
        getDeclaration()->connections.push_back(value);
        return *this;
    }
};

} // namespace raul::frontend

#endif // FRONTEND_GRAPH_H
