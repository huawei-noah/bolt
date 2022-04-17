// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef FRONTEND_NAME_H
#define FRONTEND_NAME_H

#include <string>

#define RAUL_FRONTEND_NAME_SEPARATOR "/"

namespace raul::frontend
{

/**
 * Name of declaration entities
 * Supports hierarchy paths: x/y/z
 */
struct Path
{
    /**
     * Ctor
     * Default ctor is disabled due to meaningless
     */
    Path() = delete;
    Path(const Path&) = default;
    Path(Path&&) noexcept = default;

    Path& operator=(Path other)
    {
        std::swap(storage, other.storage);
        return *this;
    }

    Path(const std::string& init) { storage.emplace_back(init); }
    Path(std::string&& init) { storage.emplace_back(init); }
    Path(const char* init) { storage.emplace_back(init); }
    Path(std::initializer_list<std::string> init)
        : storage{ init }
    {
    }

    /**
     * Return a list of name parts
     *
     * Example: /x/y/z -> {x,y,z}
     *
     * @return list of name parts
     */
    [[nodiscard]] const auto& parts() const { return storage; }

    /**
     * Returns number of name parts
     * @return number of name parts
     */
    [[nodiscard]] size_t depth() const { return storage.size(); }

    Path& operator/=(const Path& rhs)
    {
        const auto newDataSize = storage.size() + rhs.storage.size();
        const auto newCapacity = std::max(storage.capacity(), newDataSize);
        storage.reserve(newCapacity);
        storage.insert(storage.end(), rhs.storage.begin(), rhs.storage.end());
        return *this;
    }

    bool operator==(const Path& other) const { return storage == other.storage; }
    bool operator<(const Path& other) const { return storage < other.storage; }

    bool operator!=(const Path& other) const { return !(*this == other); }
    bool operator<=(const Path& other) const { return !(other < *this); }
    bool operator>(const Path& other) const { return other < *this; }
    bool operator>=(const Path& other) const { return !(*this < other); }

    /**
     * Returns the name with scope (=full path)
     * @return string
     */
    [[nodiscard]] std::string fullname(const std::string& sep = RAUL_FRONTEND_NAME_SEPARATOR) const
    {
        std::string result;

        for (auto& level : storage)
        {
            if (!result.empty())
            {
                result += sep;
            }
            result += level;
        }

        return result;
    }

    /**
     * Returns the name in string
     * @return string
     */
    [[nodiscard]] std::string str() const { return storage.back(); }

    friend std::ostream& operator<<(std::ostream& out, const Path& instance)
    {
        out << instance.str();
        return out;
    }

  private:
    std::vector<std::string> storage;
};

inline Path operator/(Path lhs, const Path& rhs)
{
    lhs /= rhs;
    return lhs;
}

inline Path operator"" _name(const char* s, std::size_t)
{
    return { s };
}

} // namespace raul::frontend

#endif // FRONTEND_NAME_H
