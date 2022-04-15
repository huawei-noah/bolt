// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef SYSTEM_ORDERED_MAP_H
#define SYSTEM_ORDERED_MAP_H

namespace raul::system
{
template<typename Key, typename T>
struct ordered_map
{
    using value_type = std::pair<Key, T>;
    using iterator = typename std::vector<value_type>::iterator;
    using const_iterator = typename std::vector<value_type>::const_iterator;
    
    ordered_map() = default;
    ordered_map(const ordered_map& other)
        : data{ other.data }
        , dictionary{ other.dictionary }
    {
    }

    T& operator[](const Key& k)
    {
        if (dictionary.find(k) == dictionary.end())
        {
            dictionary[k] = data.insert(data.end(), value_type(k, T{}));
        }
        return dictionary[k]->second;
    }
    T& operator[](size_t pos) { return data[pos].second; }

    iterator find(const Key& key) { return dictionary.find(key); }
    const_iterator find(const Key& key) const { return dictionary.find(key); }
    bool contains(const Key& key) const { return dictionary.find(key) != dictionary.end(); }

    iterator begin() noexcept { return data.begin(); }
    const_iterator begin() const noexcept { return data.begin(); }
    const_iterator cbegin() const noexcept { return data.cbegin(); }
    iterator end() noexcept { return data.end(); }
    const_iterator end() const noexcept { return data.end(); }
    const_iterator cend() const noexcept { return data.cend(); }

    void clear() noexcept
    {
        dictionary.clear();
        data.clear();
    }

    [[nodiscard]] size_t size() const noexcept { return data.size(); }
    [[nodiscard]] bool empty() const noexcept { return data.empty(); }

    std::vector<value_type> data;
    std::unordered_map<Key, iterator> dictionary;
};

template<typename Key, typename T>
auto begin(ordered_map<Key, T>& x)
{
    return x.begin();
}

template<typename Key, typename T>
auto end(ordered_map<Key, T>& x)
{
    return x.end();
}

}

#endif // SYSTEM_ORDERED_MAP_H
