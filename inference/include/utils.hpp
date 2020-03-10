// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _UTILS_HPP
#define _UTILS_HPP

#include "ut_util.h"

std::string extract_class_function(std::string&& pretty_function);
std::string extract_file_function(std::string&& pretty_function);

#define __CLASS_FUNCTION__ extract_class_function(std::string(__PRETTY_FUNCTION__))
#define __FILE_FUNCTION__ extract_file_function(std::string(__FILE__)+"::"+std::string(__FUNCTION__))

void ut_time_init();
void ut_time_tic(std::string key);
void ut_time_toc(std::string key);
void ut_time_statistics();

#define UTIL_TIME_INIT ut_time_init();
#define UTIL_TIME_TIC(str) ut_time_tic(str);
#define UTIL_TIME_TOC(str) ut_time_toc(str);
#define UTIL_TIME_STATISTICS ut_time_statistics();


#endif //_UTILS_HPP
