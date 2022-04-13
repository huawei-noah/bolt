// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef FRONTEND_PORT_H
#define FRONTEND_PORT_H

#include <iostream>
#include <optional>
#include <utility>

namespace raul::frontend
{

using Name = std::string;
struct Path;
struct Port;

using PortMap = std::pair<Port, Port>;

struct Port
{
    Port(Path layer, Name port)
        : layer{ std::move(layer) }
        , port{ std::move(port) }
    {
    }

    explicit Port(Name port)
        : layer{ std::nullopt }
        , port{ std::move(port) }
    {
    }

    [[nodiscard]] PortMap to(const Port& target) const { return { *this, target }; }
    [[nodiscard]] PortMap to(Port&& target) const { return { *this, target }; }

    [[nodiscard]] std::optional<Path> getLayer() const { return layer; }
    [[nodiscard]] Name getPort() const { return port; }
    [[nodiscard]] Path getPath() const
    {
        if (layer)
        {
            return *layer / port;
        }

        return port;
    }

    friend std::ostream& operator<<(std::ostream& out, const Port& obj)
    {
        out << "Port(" << obj.getPath().fullname("::") << ")";
        return out;
    }

    bool operator==(const Port& other) const { return (layer == other.layer) && (port == other.port); }
    bool operator<(const Port& other) const { return (layer < other.layer) && (port < other.port); }

    bool operator!=(const Port& other) const { return !(*this == other); }
    bool operator<=(const Port& other) const { return !(other < *this); }
    bool operator>(const Port& other) const { return other < *this; }
    bool operator>=(const Port& other) const { return !(*this < other); }

  private:
    std::optional<Path> layer;
    Name port;
};

} // namespace raul::frontend

#endif // FRONTEND_PORT_H
