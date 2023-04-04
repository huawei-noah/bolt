// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef NAMED_TUPLE_H
#define NAMED_TUPLE_H

#include <cstdint>
#include <tuple>

namespace
{


namespace foonathan
{
namespace string_id
{
namespace detail
{
using hash_type = std::uint64_t;

constexpr hash_type fnv_basis = 14695981039346656037ull;
constexpr hash_type fnv_prime = 109951162821ull;

// FNV-1a 64 bit hash
constexpr hash_type sid_hash(const char* str, hash_type hash = fnv_basis) noexcept
{
    return *str ? sid_hash(str + 1, (hash ^ *str) * fnv_prime) : hash;
}
}
}
}

template<typename Hash, typename... Ts>
struct named_param : public std::tuple<std::decay_t<Ts>...>
{
    using hash = Hash;

    named_param(Ts&&... ts)
        : std::tuple<std::decay_t<Ts>...>(std::forward<Ts>(ts)...){};

    template<typename P>
    named_param<Hash, P> operator[](P&& p)
    {
        return named_param<Hash, P>(std::forward<P>(p));
    };
};

template<typename Hash>
using make_named_param = named_param<Hash>;

template<typename... Params>
struct named_tuple : public std::tuple<Params...>
{

    template<typename... Args>
    named_tuple(Args&&... args)
        : std::tuple<Args...>(std::forward<Args>(args)...)
    {
    }

    static const std::size_t error = -1;

    template<std::size_t I = 0, typename Hash>
    constexpr typename std::enable_if<I == sizeof...(Params), const std::size_t>::type static get_element_index()
    {
        return error;
    }

    template<std::size_t I = 0, typename Hash>
        constexpr typename std::enable_if < I<sizeof...(Params), const std::size_t>::type static get_element_index()
    {
        using elementType = typename std::tuple_element<I, std::tuple<Params...>>::type;
        return (std::is_same<typename elementType::hash, Hash>::value) ? I : get_element_index<I + 1, Hash>();
    }

    template<typename Hash>
    auto& get()
    {
        constexpr std::size_t index = get_element_index<0, Hash>();
        static_assert((index != error), "Wrong named tuple key");
        auto& param = (std::get<index>(static_cast<std::tuple<Params...>&>(*this)));
        return std::get<0>(param);
    }

    template<typename NP>
    auto& operator[](NP&& param)
    {
        (void)param;
        return get<typename NP::hash>();
    }
};

template<typename... Args>
auto make_named_tuple(Args&&... args)
{
    return named_tuple<Args...>(std::forward<Args>(args)...);
}

} // anonymous namespace

#define param(x) (make_named_param<std::integral_constant<foonathan::string_id::detail::hash_type, foonathan::string_id::detail::sid_hash(x)>>{})

#endif