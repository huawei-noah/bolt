// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include "cnn.hpp"

// template<Arch A>
// void CNN<A>::infer_tmp_memory_size(){

// }

// template<Arch A>
// void CNN<A>::assign_tmp_tensor(){

// }

// template<Arch A>
// void CNN<A>::sort_operators() {

// }

// template<Arch A>
// EE CNN<A>::infer_output_tensors_size(Vec<TensorDesc> in_dims) {

// }

// template<Arch A>
// void CNN<A>::assign_output_tensor() {

// }


#ifdef _USE_NEON
template class CNN<ARM_A55>;
template class CNN<ARM_A76>;
#endif

#ifdef _USE_MALI
template class CNN<MALI>;
#endif
