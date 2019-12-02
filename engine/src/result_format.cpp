// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include <algorithm>
#include "result_format.hpp"

template<typename T>
Vec<int> topK_index(void *data, U32 len, U32 topK){
    Vec<int> index(len);
    for (U32 i = 0; i < index.size(); i++) {
        index[i] = i;
    }

    T* dataPtr = (T *)data;
    sort(index.begin(), index.end(),
        [&](const int& a, const int& b) {
            return (dataPtr[a] > dataPtr[b]);
        }
    );

    Vec<int>::const_iterator first = index.begin() + 0;
    Vec<int>::const_iterator last = index.begin() + topK;
    Vec<int> indexesTopK(first, last);

    return indexesTopK;
}
template Vec<int> topK_index<F32>(void *data, U32 len, U32 topK);
template Vec<int> topK_index<F16>(void *data, U32 len, U32 topK);
template Vec<int> topK_index<INT8>(void *data, U32 len, U32 topK);
