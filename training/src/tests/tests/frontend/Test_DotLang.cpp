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
#include <training/frontend/processors/DotLangPrinter.h>

using namespace raul::frontend;

namespace UT
{

void check(std::string value, std::string reference)
{
    value.erase(std::remove_if(value.begin(), value.end(), [](unsigned char x) { return std::isspace(x); }), value.end());
    reference.erase(std::remove_if(reference.begin(), reference.end(), [](unsigned char x) { return std::isspace(x); }), reference.end());
    ASSERT_STRCASEEQ(value.c_str(), reference.c_str());
}

TEST(TestDotLang, EmptyGraphUnit)
{
    std::stringstream ss;

    auto g = Graph{};

    auto dotPrinter = DotLangPrinter{};
    g.apply(dotPrinter);
    dotPrinter.print(ss);

    check(ss.str(),
          "digraph {"
          "rankdir=LR;"
          "concentrate=true;"
          "node[shape=record];"
          "}");
}

TEST(TestDotLang, LayerUnit)
{
    std::stringstream ss;

    auto g = Linear{ 3 };
    auto dotPrinter = DotLangPrinter{};
    g.apply(dotPrinter);
    dotPrinter.print(ss);

    check(ss.str(),
          "digraph {"
          "rankdir=LR;"
          "concentrate=true;"
          "node[shape=record];"
          "element[label=\"noname:Linear|{<in>in|<out>out}\"];"
          "}");
}

TEST(TestDotLang, OneElementGraphUnit)
{
    std::stringstream ss;

    auto g = Graph{ Linear{ 3 } };
    auto dotPrinter = DotLangPrinter{};
    g.apply(dotPrinter);
    dotPrinter.print(ss);

    check(ss.str(),
          "digraph {"
          "rankdir=LR;"
          "concentrate=true;"
          "node[shape=record];"
          "subgraph cluster {"
          "element_0[label=\"0:Linear|{<in>in|<out>out}\"];"
          "port_in[label=\"in\" shape=oval];"
          "port_out[label=\"out\" shape=oval];"
          "port_in->element_0:in;"
          "element_0:out->port_out;"
          "}"
          "}");
}

TEST(TestDotLang, TwoElementsGraphUnit)
{
    std::stringstream ss;

    auto g = Graph{ Linear{ 3 }, Linear{ 8 } };
    auto dotPrinter = DotLangPrinter{};
    g.apply(dotPrinter);
    dotPrinter.print(ss);

    check(ss.str(),
          "digraph {"
          "rankdir=LR;"
          "concentrate=true;"
          "node[shape=record];"
          "subgraph cluster"
          "{"
          "element_0[label=\"0:Linear|{<in>in|<out>out}\"];"
          "element_1[label=\"1:Linear|{<in>in|<out>out}\"];"
          "port_in[label=\"in\" shape=oval];"
          "port_out[label=\"out\" shape=oval];"
          "port_in->element_0:in;"
          "element_0:out->element_1:in;"
          "element_1:out->port_out;"
          "}"
          "}");
}

TEST(TestDotLang, NestedGraphUnit)
{
    std::stringstream ss;

    auto g = Graph{ { Graph{ Linear{ 3 } } } };
    auto dotPrinter = DotLangPrinter{};
    g.apply(dotPrinter);
    dotPrinter.print(ss);

    check(ss.str(),
          "digraph {"
          "rankdir=LR;"
          "concentrate=true;"
          "node[shape=record];"
          "subgraph cluster"
          "{"
          "subgraph cluster_0"
          "{"
          "label=\"0\""
          "element_0_0[label=\"0:Linear|{<in>in|<out>out}\"];"
          "port_0_in[label=\"in\" shape=oval];"
          "port_0_out[label=\"out\" shape=oval];"
          "port_0_in->element_0_0:in;"
          "element_0_0:out->port_0_out;"
          "}"
          "port_in[label=\"in\" shape=oval];"
          "port_out[label=\"out\" shape=oval];"
          "port_in->port_0_in;"
          "port_0_out->port_out;"
          "}"
          "}");
}

TEST(TestDotLang, TwoNestedGraphsUnit)
{
    std::stringstream ss;

    auto g = Graph{ Graph{ Linear{ 3 }, Linear{ 4 } }, Graph{ Linear{ 5 }, Linear{ 6 } } };
    auto dotPrinter = DotLangPrinter{};
    g.apply(dotPrinter);
    dotPrinter.print(ss);

    check(ss.str(),
          "digraph {"
          "rankdir=LR;"
          "concentrate=true;"
          "node[shape=record];"
          "subgraph cluster"
          "{"
          "subgraph cluster_0"
          "{"
          "label=\"0\""
          "element_0_0[label=\"0:Linear|{<in>in|<out>out}\"];"
          "element_0_1[label=\"1:Linear|{<in>in|<out>out}\"];"
          "port_0_in[label=\"in\" shape=oval];"
          "port_0_out[label=\"out\" shape=oval];"
          "port_0_in->element_0_0:in;"
          "element_0_0:out->element_0_1:in;"
          "element_0_1:out->port_0_out;"
          "}"
          "subgraph cluster_1"
          "{"
          "label=\"1\""
          "element_1_0[label=\"0:Linear|{<in>in|<out>out}\"];"
          "element_1_1[label=\"1:Linear|{<in>in|<out>out}\"];"
          "port_1_in[label=\"in\" shape=oval];"
          "port_1_out[label=\"out\" shape=oval];"
          "port_1_in->element_1_0:in;"
          "element_1_0:out->element_1_1:in;"
          "element_1_1:out->port_1_out;"
          "}"
          "port_in[label=\"in\" shape=oval];"
          "port_out[label=\"out\" shape=oval];"
          "port_in->port_0_in;"
          "port_0_out->port_1_in;"
          "port_1_out->port_out;"
          "}"
          "}");
}

TEST(TestDotLang, TwoNamedNestedGraphsUnit)
{
    std::stringstream ss;

    auto g = Graph{ { "x", Graph{ { "l1", Linear{ 3 } }, Linear{ 4 } } }, { "y", Graph{ { "l2", Linear{ 5 } }, Linear{ 6 } } } };
    auto dotPrinter = DotLangPrinter{};
    g.apply(dotPrinter);
    dotPrinter.print(ss);

    check(ss.str(),
          "digraph {"
          "rankdir=LR;"
          "concentrate=true;"
          "node[shape=record];"
          "subgraph cluster"
          "{"
          "subgraph cluster_x"
          "{"
          "label=\"x\""
          "element_x_l1[label=\"l1:Linear|{<in>in|<out>out}\"];"
          "element_x_1[label=\"1:Linear|{<in>in|<out>out}\"];"
          "port_x_in[label=\"in\" shape=oval];"
          "port_x_out[label=\"out\" shape=oval];"
          "port_x_in->element_x_l1:in;"
          "element_x_l1:out->element_x_1:in;"
          "element_x_1:out->port_x_out;"
          "}"
          "subgraph cluster_y"
          "{"
          "label=\"y\""
          "element_y_l2[label=\"l2:Linear|{<in>in|<out>out}\"];"
          "element_y_1[label=\"1:Linear|{<in>in|<out>out}\"];"
          "port_y_in[label=\"in\" shape=oval];"
          "port_y_out[label=\"out\" shape=oval];"
          "port_y_in->element_y_l2:in;"
          "element_y_l2:out->element_y_1:in;"
          "element_y_1:out->port_y_out;"
          "}"
          "port_in[label=\"in\" shape=oval];"
          "port_out[label=\"out\" shape=oval];"
          "port_in->port_x_in;"
          "port_x_out->port_y_in;"
          "port_y_out->port_out;"
          "}"
          "}");
}

TEST(TestDotLang, TwoNamedNestedGraphsAndSkipUnit)
{
    std::stringstream ss;

    auto subGraph1 = Graph{ { "l1", Linear{ 3 } }, { "l2", Linear{ 4 } } };
    auto subGraph2 = Graph{ { "l3", Linear{ 5 } }, { "l4", Linear{ 6 } } };
    auto g = Graph{ { { "g1", subGraph1 }, { "g2", subGraph2 } },
                    { Port("in").to(Port("g1", "in")), Port("g1", "out").to(Port("g2", "in")), Port("g2", "out").to(Port("out")), Port("g1", "out").to(Port("out_aux")) } };

    auto dotPrinter = DotLangPrinter{};
    g.apply(dotPrinter);
    dotPrinter.print(ss);

    check(ss.str(),
          "digraph {"
          "rankdir=LR;"
          "concentrate=true;"
          "node[shape=record];"
          "subgraph cluster"
          "{"
          "subgraph cluster_g1"
          "{"
          "label=\"g1\""
          "element_g1_l1[label=\"l1:Linear|{<in>in|<out>out}\"];"
          "element_g1_l2[label=\"l2:Linear|{<in>in|<out>out}\"];"
          "port_g1_in[label=\"in\" shape=oval];"
          "port_g1_out[label=\"out\" shape=oval];"
          "port_g1_in->element_g1_l1:in;"
          "element_g1_l1:out->element_g1_l2:in;"
          "element_g1_l2:out->port_g1_out;"
          "}"
          "subgraph cluster_g2"
          "{"
          "label=\"g2\""
          "element_g2_l3[label=\"l3:Linear|{<in>in|<out>out}\"];"
          "element_g2_l4[label=\"l4:Linear|{<in>in|<out>out}\"];"
          "port_g2_in[label=\"in\" shape=oval];"
          "port_g2_out[label=\"out\" shape=oval];"
          "port_g2_in->element_g2_l3:in;"
          "element_g2_l3:out->element_g2_l4:in;"
          "element_g2_l4:out->port_g2_out;"
          "}"
          "port_in[label=\"in\" shape=oval];"
          "port_out[label=\"out\" shape=oval];"
          "port_out_aux[label=\"out_aux\" shape=oval];"
          "port_in->port_g1_in;"
          "port_g1_out->port_g2_in;"
          "port_g2_out->port_out;"
          "port_g1_out->port_out_aux;"
          "}"
          "}");
}

} // UT namespace