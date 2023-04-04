// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef NAME_GENERATOR_H
#define NAME_GENERATOR_H

#include <random>
#include <string>
#include <utility>

#include <training/base/common/Conversions.h>

#define DEFAULT_COUNTER_VALUE 1U
#define DEFAULT_RANDOM_PREFIX_LENGTH 10U
#define ASCII_FROM 97U
#define ASCII_TO 122U

namespace raul
{
/**
 * @brief Name generator with an incremental suffix
 *
 * The generator produces name with fixed prefix and incremental suffix
 * in the following manner: prefixNN, where NN is an integer.
 *
 * The generator uses a random prefix if it is not specified.
 */
class NameGenerator
{
  public:
    /**
     * @brief Get random string of a specific length
     *
     * @param Length Length of the random string
     *
     * This function uses uniform distribution for generating random sequence of characters
     * withing the range [97, 122]. This range can be redefined using the fillowing macro
     * [ASCII_FROM, ASCII_TO].
     */
    static std::string getRandomString(const unsigned int length = DEFAULT_RANDOM_PREFIX_LENGTH)
    {
        std::random_device random_device;
        std::mt19937 generator(random_device());
        std::uniform_int_distribution<> distribution(ASCII_FROM, ASCII_TO);

        std::string random_string;
        random_string.reserve(length);

        for (size_t i = 0; i < length; ++i)
        {
            random_string += static_cast<char>(distribution(generator));
        }

        return random_string;
    }

    /**
     * @brief Construct a new Name Generator object with a specific prefix
     *
     * @param prefix Prefix string
     */
    explicit NameGenerator(std::string prefix)
        : mPrefix(std::move(prefix))
        , mNext(DEFAULT_COUNTER_VALUE)
    {
    }
    /**
     * @brief Construct a new Name Generator object with a random prefix specified by the length
     *
     * @param length Length of the random prefix
     */
    explicit NameGenerator(const unsigned int length)
        : mPrefix(getRandomString(length))
        , mNext(DEFAULT_COUNTER_VALUE)
    {
    }
    /**
     * @brief Construct a new Name Generator object with a random prefix of length 10
     *
     * Default values of the length can be redefined, see DEFAULT_RANDOM_PREFIX_LENGTH macro.
     */
    NameGenerator()
        : mPrefix(getRandomString(DEFAULT_RANDOM_PREFIX_LENGTH))
        , mNext(DEFAULT_COUNTER_VALUE)
    {
    }
    ~NameGenerator(){}
    /**
     * @brief Generate the next name with prefix and incremented suffix
     *
     * The generator starts from 1 (can be redefined, see DEFAULT_COUNTER_VALUE). The counter can be reset by reset() method.
     *
     * @return std::string
     */
    std::string generate() { return mPrefix + Conversions::toString(mNext++); }

    /**
     * @brief Set the internal counter value to 1 (can be redefined, see DEFAULT_COUNTER_VALUE).
     *
     */
    void reset() { mNext = DEFAULT_COUNTER_VALUE; }

    /**
     * @brief Set the internal counter specific value.
     *
     * @param next New counter value
     */
    void setNext(size_t next) { mNext = next; }

    /**
     * @brief Get the next counter value
     *
     * @return size_t Current counter value
     */
    size_t getNext() const { return mNext; }

    /**
     * @brief Get the prefix
     *
     * @return std::string Prefix string
     */
    std::string getPrefix() const { return mPrefix; }

    /**
     * @brief Set the prefix
     *
     * @param prefix New prefix
     */

    void setPrefix(const std::string& prefix) { mPrefix = prefix; }

  private:
    std::string mPrefix;
    size_t mNext;
};

} // raul namespace

#endif
