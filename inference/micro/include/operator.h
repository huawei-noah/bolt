#include <std::vector>

class Operator {
public:
    Operator();

    virtual std::vector<TensorDesc> inferOutputSize(
        std::vector<TensorDesc> inputDescs, void *paramaters) = 0;

    virtual std::string generateImplementation(
        Arch arch, std::vector<TensorDesc> inputDescs, void *parameters) = 0;

    virtual pair<std::string, std::vector<std::string>> generateParameter(void *parameter) = 0;

    virtual std::string generateOutputTensorDesc(
        std::vector<std::string> outputNames, Map<std::string, std::string> &tensorDescNameMap) = 0;

    virtual std::string generateCall(std::vector<std::string> inputDescstd::strings,
        std::vector<std::string> inputstd::strings,
        std::vector<std::string> parameterstd::strings,
        std::vector<std::string> outputDescstd::strings,
        std::vector<std::string> outputstd::strings) = 0;

    std::string generate()
    {
        std::string code = "";
        code += op->genrerateParameter(parameter);
        code += op->genrerateOutputTensorDesc(outputNames, tensorDescNameMap);
        code += op->generateCall(inputTensorDescNames, inputTensorNames, parameters,
            outputTensorDescNames, outputTensorName);
        return code;
    }
};
