// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include <string>
#include <map>
#include <vector>
#include <iostream>
#include <algorithm>

#include "utils.hpp"


std::string extract_class_function(std::string&& pretty_function) {
    auto pos = pretty_function.find('(');
    if (pos != std::string::npos)
        pretty_function.erase(pretty_function.begin()+pos, pretty_function.end());

    pos = pretty_function.rfind(' ');
    if (pos != std::string::npos)
        pretty_function.erase(pretty_function.begin(), pretty_function.begin()+pos+1);

    return std::move(pretty_function);
}


std::string extract_file_function(std::string&& pretty_function) {
    auto pos = pretty_function.find('(');
    if (pos != std::string::npos)
        pretty_function.erase(pretty_function.begin()+pos, pretty_function.end());

    pos = pretty_function.rfind('/');
    if (pos != std::string::npos)
        pretty_function.erase(pretty_function.begin(), pretty_function.begin()+pos+1);

    return std::move(pretty_function);
}


std::map<std::string, double> time_tic;
std::map<std::string, double> time_toc;
std::map<std::string, double> time_statistics;


void ut_time_init() {
    time_tic.clear();
    time_toc.clear();
    time_statistics.clear();
}


void ut_time_tic(std::string key) {
   double value = ut_time_ms();
   time_tic[key] = value;
}


void ut_time_toc(std::string key) {
   double value = ut_time_ms();
   time_toc[key] = value;
   std::map<std::string, double>::iterator iter = time_tic.find(key);
   if (iter == time_tic.end())
       std::cout << "[WARNING] mismatched UTIL_TIME_TIC/UTIL_TIME_TOC " << key << std::endl;
   else {
       iter = time_statistics.find(key);
       if(iter == time_statistics.end())
           time_statistics[key] = time_toc[key] - time_tic[key];
       else
           time_statistics[key] += time_toc[key] - time_tic[key];
   }
}

void ut_time_statistics() {
    std::vector<std::pair<std::string, double>> vec(time_statistics.begin(), time_statistics.end());

    sort(vec.begin(), vec.end(),
        [&](const std::pair<std::string, double>& a, const std::pair<std::string, double>& b) {
            return (a.second > b.second);
        }
    );
    std::cout << "[TIME]" << std::endl;
    std::cout << "function\ttime" << std::endl;
    for (U32 i = 0; i < vec.size(); ++i)
        std::cout << vec[i].first << "  " << vec[i].second  << " ms"<< std::endl;
}
