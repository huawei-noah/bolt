// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef FRONTEND_GENERATOR_H
#define FRONTEND_GENERATOR_H

#include <utility>

#include "Types.h"
#include "processors/Processor.h"

namespace raul::frontend
{

struct Generator
{
    [[nodiscard]] Type getType() const { return declaration->type; }
    [[nodiscard]] const Inputs& getInputs() const { return declaration->inputs; }
    [[nodiscard]] const Outputs& getOutputs() const { return declaration->outputs; }

    void apply(Processor& proc) { handler(proc, std::nullopt); }
    void apply(Processor& proc) const { handler(proc, std::nullopt); }
    void apply(Processor& proc, Path position) { handler(proc, std::move(position)); }
    void apply(Processor& proc, Path position) const { handler(proc, std::move(position)); }

  protected:
    Ref<Declaration> declaration;
    Handler handler;
};

template<class T>
struct GeneratorTyped : Generator
{
    template<typename...>
    struct typelist;

    template<class... Args, typename = std::enable_if_t<!std::is_same_v<typelist<GeneratorTyped>, typelist<std::decay_t<Args>...>>>>
    explicit GeneratorTyped(Args&&... args)
    {
        declaration = std::make_shared<T>(args...);
        handler = [declaration = getDeclaration()](Processor& proc, std::optional<Path> position) { proc.process(*declaration, position); };
    }

  protected:
    [[nodiscard]] Ref<T> getDeclaration() { return std::static_pointer_cast<T>(declaration); };
};
}

#endif // FRONTEND_GENERATOR_H
