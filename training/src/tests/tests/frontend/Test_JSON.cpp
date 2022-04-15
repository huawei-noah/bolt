// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <tests/tools/TestTools.h>

#include <training/frontend/Graph.h>
#include <training/frontend/Layers.h>
#include <training/frontend/io/JSON.h>
#include <training/frontend/processors/TextPrinter.h>

using namespace raul::frontend;

namespace UT
{

TEST(TestJSON, LinearUnit)
{

    auto data = R"(
                    {
                        "type": "linear",
                        "features": 20,
                        "bias": true
                    }
                )"_json;

    auto g = frontend::io::fromJSON(data);

    std::stringstream ss;
    TextPrinter printer(ss);
    g.apply(printer);

    ASSERT_STREQ(ss.str().c_str(), "Linear(features=20, bias=true)");
}

TEST(TestJSON, EmptyGraphUnit)
{

    auto data = R"(
                   {
                        "type": "graph",
                        "nodes": {}
                   }
                )"_json;

    auto g = frontend::io::fromJSON(data);

    std::stringstream ss;
    TextPrinter printer(ss);
    g.apply(printer);

    ASSERT_STREQ(ss.str().c_str(), "[]");
}

TEST(TestJSON, NodeOnlyGraphUnit)
{

    auto data = R"(
                    {
                       "type": "graph",
                       "nodes": {
                            "A": {"type": "linear", "features": 1},
                            "B": {"type": "linear", "features": 2}
                        }
                    }
                )"_json;

    auto g = frontend::io::fromJSON(data);

    std::stringstream ss;
    TextPrinter printer(ss);
    g.apply(printer);

    ASSERT_STREQ(ss.str().c_str(), "[Linear(name=A, features=1, bias=false),Linear(name=B, features=2, bias=false)]");
}

TEST(TestJSON, NodeOnlyNestedGraphUnit)
{

    auto data = R"(
                    {
                       "type": "graph",
                       "nodes": {
                            "A": {
                                "type": "graph",
                                "nodes": {
                                    "Inner": {"type": "linear", "features": 3}
                                }
                            },
                            "B": {"type": "linear", "features": 2}
                        }
                    }
                )"_json;

    auto g = frontend::io::fromJSON(data);

    std::stringstream ss;
    TextPrinter printer(ss);
    g.apply(printer);

    ASSERT_STREQ(ss.str().c_str(), "[A:[Linear(name=Inner, features=3, bias=false)],Linear(name=B, features=2, bias=false)]");
}

TEST(TestJSON, NodePortsGraphUnit)
{

    auto data = R"(
                    {
                       "type": "graph",
                       "nodes": {
                            "A": {"type": "linear", "features": 1},
                            "B": {"type": "linear", "features": 2}
                        },
                        "edges": [
                            {
                                "from": {"port": "in"},
                                "to": {"layer": "A", "port": "in"}
                            },
                            {
                                "from": {"layer": "B", "port": "out"},
                                "to": {"port": "out"}
                            }
                        ]
                    }
                )"_json;

    auto g = frontend::io::fromJSON(data);

    std::stringstream ss;
    TextPrinter printer(ss);
    g.apply(printer);

    ASSERT_STREQ(ss.str().c_str(), "[Linear(name=A, features=1, bias=false),Linear(name=B, features=2, bias=false)|Port(in)->Port(A::in),Port(B::out)->Port(out)]");
}

} // UT namespace