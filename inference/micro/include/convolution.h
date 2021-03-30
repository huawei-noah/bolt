#include "operator.h"

class Convolution ::Operator {
public:
    // use tensor_computing interface or rewrite
    std::vector<TensorDesc> inferOutputSize(
        std::vector<TensorDesc> inputDescs, void *paramaters) override
    {}

    std::string generateImplementation(
        Arch arch, std::vector<TensorDesc> inputDescs, void *parameters) override
    {}

    pair<std::string, std::vector<std::string>> generateParameter(void *parameter) override
    {}

    std::string generateOutputTensorDesc(std::vector<std::string> outputNames,
        Map<std::string, std::string> &tensorDescNameMap) override
    {}
    // output = "convolution(inputDesc, input, params, outputDesc, output);"
    std::string generateCall(std::vector<std::string> inputDescstd::strings,
        std::vector<std::string> inputstd::strings,
        std::vector<std::string> parameterstd::strings,
        std::vector<std::string> outputDescstd::strings,
        std::vector<std::string> outputstd::strings) override
    {}
};
