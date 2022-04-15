// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef TENSOR_NAME_H
#define TENSOR_NAME_H

#include <iostream>
#include <set>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#define TENSOR_SEP "::"
#define TENSOR_GRADIENT_POSTFIX "Gradient"

namespace raul
{

/**
 * @brief Wrapper for names of entities
 */
struct Name
{
    Name() = default;
    ~Name() = default;

    Name(const std::string& str)
        : string(str)
    {
    }
    Name(const char* str)
        : string(str)
    {
    }

    Name(const Name& other)
        : string(other.string)
    {
    }

    Name& operator=(Name other)
    {
        std::swap(string, other.string);
        return *this;
    }

    operator std::string() const { return string; }
    operator std::string_view() const { return std::string_view(string); }

    Name& operator+=(const Name& rhs)
    {
        this->string += rhs.string;
        return *this;
    }

    Name& operator/=(const Name& rhs)
    {
        if (!rhs.empty())
        {
            if (!empty())
            {
                this->string += std::string(TENSOR_SEP) + rhs.string;
            }
            else
            {
                this->string = rhs.string;
            }
        }
        return *this;
    }

    /**
     * @brief Get prefix string before last TENSOR_SEP
     *
     */
    [[nodiscard]] Name getPrefix() const;

    /**
     * @brief Get last string after last TENSOR_SEP
     *
     */
    [[nodiscard]] Name getLastName() const;

    [[nodiscard]] std::string grad() const { return string + TENSOR_GRADIENT_POSTFIX; }

    [[nodiscard]] size_t size() const { return string.size(); }
    [[nodiscard]] bool empty() const { return string.empty(); }
    [[nodiscard]] const char* c_str() const { return string.c_str(); }
    [[nodiscard]] std::string str() const { return string; }

    bool operator==(const Name& other) const { return string == other.string; }
    bool operator<(const Name& other) const { return string < other.string; }

    bool operator!=(const Name& other) const { return !(*this == other); }
    bool operator<=(const Name& other) const { return !(other < *this); }
    bool operator>(const Name& other) const { return other < *this; }
    bool operator>=(const Name& other) const { return !(*this < other); }

    friend std::ostream& operator<<(std::ostream& out, const Name& instance) { return instance.as_ostream(out); }

  private:
    std::ostream& as_ostream(std::ostream& out) const
    {
        out << string;
        return out;
    }
    std::string string;
};

inline Name operator+(Name lhs, const Name& rhs)
{
    lhs += rhs;
    return lhs;
}

inline Name operator/(Name lhs, const Name& rhs)
{
    lhs /= rhs;
    return lhs;
}

typedef std::vector<Name> Names;
typedef std::set<Name> NameSet;
typedef std::unordered_set<Name> NameUnorderedSet;

} // namespace raul

namespace std
{

template<>
struct hash<raul::Name>
{
    std::size_t operator()(const raul::Name& k) const { return hash<std::string>()(k.str()); }
};

}

#endif // TENSOR_NAME_H
