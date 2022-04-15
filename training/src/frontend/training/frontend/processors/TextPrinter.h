// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef FRONTEND_PROCESSOR_PRINTER_H
#define FRONTEND_PROCESSOR_PRINTER_H

#include "Processor.h"

namespace raul::frontend
{

struct TextPrinter : Processor
{
    TextPrinter() = default;
    explicit TextPrinter(std::ostream& init)
        : stream{ init }
    {
    }

    TextPrinter useIndent(size_t indent = 2)
    {
        indentFlag = true;
        indentStep = indent;
        return *this;
    }

    TextPrinter showAddress()
    {
        addressFlag = true;
        return *this;
    }

  private:
    [[nodiscard]] static auto printBool(bool x) { return x ? "true" : "false"; }
    [[nodiscard]] static auto printName(const std::optional<Path>& x, const std::string& prefix, const std::string& suffix)
    {
        std::stringstream ss;
        if (x)
        {
            ss << prefix << *x << suffix;
        }

        return ss.str();
    }

    [[nodiscard]] auto printAddress(const Declaration& declaration, const std::string& suffix) const
    {
        std::stringstream ss;
        if (addressFlag)
        {
            ss << "address=" << &declaration << suffix;
        }
        return ss.str();
    }

    [[nodiscard]] auto printIndent() const { return indentFlag ? std::string(level * indentStep, ' ') : ""s; }
    [[nodiscard]] auto printNewLine() const { return indentFlag ? "\n" : ""; }

    void printGraphPrelude(const std::optional<Path>& name, const GraphDeclaration& declaration) const
    {
        stream << printIndent() << printName(name, "", ":") << "[" << printAddress(declaration, "") << printNewLine();
    }
    void printGraphInterlude() const
    {
        if (indentFlag)
        {
            stream << printIndent() << "----" << printNewLine();
        }
        else
        {
            stream << "|";
        }
    }
    void printGraphPostlude() const { stream << printIndent() << "]"; }

    void indentStepIn()
    {
        if (indentFlag)
        {
            ++level;
        }
    }

    void indentStepOut()
    {
        if (indentFlag)
        {
            --level;
        }
    }

  public:
    void process(const LinearDeclaration& x, const std::optional<Path> name) override
    {
        stream << printIndent() << "Linear(" << printAddress(x, ", ") << printName(name, "name=", ", ") << "features=" << x.features << ", bias=" << printBool(x.bias) << ")";
    }
    void process(const ReLUDeclaration& x, const std::optional<Path> name) override { stream << printIndent() << "ReLU(" << printAddress(x, "") << printName(name, "name=", "") << ")"; }
    void process(const SigmoidDeclaration& x, const std::optional<Path> name) override { stream << printIndent() << "Sigmoid(" << printAddress(x, "") << printName(name, "name=", "") << ")"; }
    void process(const TanhDeclaration& x, const std::optional<Path> name) override { stream << printIndent() << "Tanh(" << printAddress(x, "") << printName(name, "name=", "") << ")"; }
    void process(const SoftmaxDeclaration& x, const std::optional<Path> name) override { stream << printIndent() << "Softmax(" << printAddress(x, "") << printName(name, "name=", "") << ")"; }
    void process(const DropoutDeclaration& x, const std::optional<Path> name) override { stream << printIndent() << "Dropout(" << printAddress(x, "") << printName(name, "name=", "") << ")"; }
    void process(const LambdaDeclaration& x, const std::optional<Path> name) override { stream << printIndent() << "Lambda(" << printAddress(x, "") << printName(name, "name=", "") << ")"; }
    void process(const ReshapeDeclaration& x, const std::optional<Path> name) override
    {
        stream << printIndent() << "Reshape(" << printAddress(x, "") << printName(name, "name=", ", ");
        stream << "[";
        for (auto it = x.shape.cbegin(); it != x.shape.cend(); ++it)
        {
            stream << *it;
            if (std::distance(it, x.shape.cend()) > 1)
            {
                stream << ",";
            }
        }
        stream << "]";
        stream << ")";
    }
    void process(const GraphDeclaration& x, const std::optional<Path> rootPath) override
    {
        printGraphPrelude(rootPath, x);
        for (auto it = x.elements.begin(); it != x.elements.end(); ++it)
        {
            auto& [elementName, elementGenerator] = *it;
            auto path = rootPath ? *rootPath / elementName : elementName;
            indentStepIn();
            elementGenerator.apply(*this, path);
            indentStepOut();
            if (std::distance(it, x.elements.end()) > 1)
            {
                stream << ",";
            }
            stream << printNewLine();
        }
        indentStepIn();
        if (!x.connections.empty())
        {
            printGraphInterlude();
        }
        for (auto it = x.connections.begin(); it != x.connections.end(); ++it)
        {
            auto& [from, to] = *it;
            stream << printIndent() << from << "->" << to;
            if (std::distance(it, x.connections.end()) > 1)
            {
                stream << ",";
            }
            stream << printNewLine();
        }
        indentStepOut();
        printGraphPostlude();
    }

    void process(const Declaration&, const std::optional<Path> name) override { stream << printIndent() << "Unknown(" << printName(name, "name=", "") << ")"; }

  private:
    std::ostream& stream = std::cout;
    bool indentFlag = false;
    bool addressFlag = false;
    size_t indentStep = 2;
    size_t level = 0;
};

} // namespace raul::frontend

#endif // FRONTEND_PROCESSOR_PRINTER_H
