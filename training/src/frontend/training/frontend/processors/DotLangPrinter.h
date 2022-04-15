// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef FRONTEND_DOT_LANG_PRINTER_H
#define FRONTEND_DOT_LANG_PRINTER_H

#include "Processor.h"

#include <iostream>
#include <optional>

#include <training/system/ordered_map.h>

namespace raul::frontend
{

struct DotLangPrinter : Processor
{
  private:
    auto dotDigraphConfig() const
    {
        std::stringstream ss;
        ss << "rankdir=LR;" << end;
        ss << "concentrate=true;" << end;
        ss << "node[shape=record];" << end;
        return ss.str();
    }

    auto portsDeclarations(const PortNames& ports) const
    {
        std::stringstream ss;
        for (auto it = ports.begin(); it != ports.end(); ++it)
        {
            ss << "<" << *it << ">" << *it;
            if (std::distance(it, ports.end()) > 1)
            {
                ss << "|";
            }
        }
        return ss.str();
    }

    auto nameString(const std::optional<Path>& path, const std::string& prefix) const
    {
        std::stringstream ss;
        ss << prefix;
        if (path)
        {
            if (!prefix.empty())
            {
                ss << "_";
            }
            ss << path->fullname("_");
        }
        return ss.str();
    }

    auto elementString(const std::optional<Path>& path, const Type type, const PortNames& inputs, const PortNames& outputs) const
    {
        std::stringstream ss;
        ss << nameString(path, "element");
        ss << "[";
        ss << "label="
           << "\"";
        if (path)
        {
            ss << *path << ":" << type << "|";
        }
        else
        {
            ss << "noname:" << type << "|";
        }
        ss << "{";
        ss << portsDeclarations(inputs);
        if (!(inputs.empty() || outputs.empty()))
        {
            ss << "|";
        }
        ss << portsDeclarations(outputs);
        ss << "}"
           << "\"";
        ss << "];" << end;
        return ss.str();
    }

    static Port getPort(const std::optional<Path>& path, const Port& port)
    {
        auto layer = port.getLayer();

        if (path && layer)
        {
            return { *path / *layer, port.getPort() };
        }

        if (path)
        {
            return { *path, port.getPort() };
        }

        if (layer)
        {
            return { *layer, port.getPort() };
        }

        return Port{ "port_" + port.getPort() };
    }

    auto portString(const std::optional<Path>& path, const std::set<Path>& subgraphs, const Port& port)
    {
        std::stringstream ss;

        if (port.getLayer())
        {
            const auto elementPath = path ? *path / *port.getLayer() : *port.getLayer();
            if(subgraphs.find(*port.getLayer()) == subgraphs.end())
            {
                ss << nameString(elementPath, "element") << ":" << port.getPort();
            }
            else
            {
                const auto p = port.getPath();
                const auto portPath = path ? *path / p : p;
                ss << nameString(portPath, "port");
            }
        }
        else
        {
            const auto p = port.getPath();
            const auto portPath = path ? *path / p : p;
            ss << nameString(portPath, "port");
        }

        return ss.str();
    }

    auto connectionString(const std::optional<Path>& path, const std::set<Path>& subgraphs, const Port& from, const Port& to)
    {
        std::stringstream ss;

        ss << portString(path, subgraphs, from);
        ss << "->";
        ss << portString(path, subgraphs, to);
        ss << ";";
        ss << end;

        return ss.str();
    }

    auto portsString(const std::optional<Path>& path, const std::vector<PortMap>& connections)
    {
        std::stringstream ss;

        std::set<Name> ports;
        auto insertExternalPort = [&](const auto port)
        {
            if (!port.getLayer())
            {
                ports.insert(port.getPort());
            }
        };

        for (const auto& [from, to] : connections)
        {
            insertExternalPort(from);
            insertExternalPort(to);
        }

        for (const auto& port : ports)
        {
            const auto portPath = path ? *path / port : Path(port);
            ss << nameString(portPath, "port") << "[label=\"" << port << "\" shape=oval];" << end;
        }

        return ss.str();
    }

    auto connectionsString(const std::optional<Path>& path, const std::vector<PortMap>& connections, const system::ordered_map<Name, Generator>& elements)
    {
        std::stringstream ss;

        std::set<Path> subgraphs;

        for (auto& [elementName, elementGenerator] : elements)
        {
            if (elementGenerator.getType() == Type::Graph)
            {
                subgraphs.insert(elementName);
            }
        }

        for (const auto& [from, to] : connections)
        {
            ss << connectionString(path, subgraphs, from, to);
        }

        return ss.str();
    }

  public:
    void print(std::ostream& out = std::cout) const
    {
        out << "digraph {" << end;
        out << dotDigraphConfig();
        out << stream.str();
        out << "}" << end;
    }

    void process(const GraphDeclaration& x, const std::optional<Path> path) override
    {
        if (x.elements.empty())
        {
            return;
        }

        stream << "subgraph " << nameString(path, "cluster") << end;
        stream << "{" << end;
        if (path)
        {
            stream << "label="
                   << "\"" << *path << "\"" << end;
        }

        for (auto& [elementName, elementGenerator] : x.elements)
        {
            auto fullName = path ? *path / elementName : elementName;
            elementGenerator.apply(*this, fullName);
        }

        stream << portsString(path, x.connections);
        stream << connectionsString(path, x.connections, x.elements);

        stream << "}" << end;
    }

    void process(const LinearDeclaration& x, const std::optional<Path> path) override { stream << elementString(path, x.type, x.inputs, x.outputs); }

  private:
    std::stringstream stream;
    std::string end = "\n";
    std::set<Path> objects;
};

} // namespace raul::frontend

#endif // FRONTEND_DOT_LANG_PRINTER_H
