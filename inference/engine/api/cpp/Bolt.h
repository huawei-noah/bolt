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

class Bolt {
public:
    Bolt();

    ~Bolt();

    std::string convert(std::string modelDirectory, std::string modelName, std::string precision);

    int set_num_threads(int threads);

    int load(std::string boltModelPath, std::string affinity = "CPU");

    std::map<std::string, std::vector<int>> get_input_info();

    std::map<std::string, std::vector<int>> get_output_info();

    std::map<std::string, std::vector<float>> infer(
        const std::map<std::string, std::vector<float>> &input);

private:
    void *boltHandle;
    void *resultHandle;

    int inputNum;
    char **inputName;
    int *inputN;
    int *inputC;
    int *inputH;
    int *inputW;
    void *inputDT;
    void *inputDF;
    void **inputData;

    int outputNum;
    char **outputName;
    int *outputN;
    int *outputC;
    int *outputH;
    int *outputW;
    void *outputDT;
    void *outputDF;
    void **outputData;
};
