#include "util.h"
#include "model.h"
#include "operator.h"
#include "test.h"
#include "compile.h"
#include "convolution.h"

#include <set>
#include <vector>
#include <string>

class Engine {
public:
    Engine(std::string engineFilePath);

    void generateOperators()
    {
        std::set<OperatorType> operatorList;
        for (auto op : ops) {
            operatorList.insert(op.type);
        }
        for (auto op : operatorList) {
            Operator *op = new Convolution();
            this->tensorDescMap[op_output] = op->infer_output_size(op_input_descs, parameter);
            std::string convolutionCodePath = op->generateOperatorImplementation(arch);
            this->sourcesFiles.append(convolutionCodePath);
        }
    }

    std::string generateEngineHeader();

    void generateMemorySegment()
    {
        // input: this->tensorDescMap�����е�desc��map
    }

    std::string generateEngineInterface();

    std::string generateEngineLogic(std::string engineFilePath)
    {
        std::string code = "";
        Operator *op;
        for (auto op : ops) {
            switch (op.type) {
                case OT_Conv: {
                    op = new Convolution();
                    break;
                }
            }
            code += op->generate();
        }
        return code;
    }

    void generateEngine()
    {
        std::string code = "";
        code += generateMemorySegment;
        code += generateEngineInterface;
        code += "\n{" code += generateEngineLogic;
        code += "}";
        writeEngineToFile;
    }

    void generate(Arch arch)
    {
        generateLog;

        parseBoltModel;
        generateModelWeightToArray;

        generateOperators(arch);

        generateEngine;
    }

private:
    Map<std::string, TensorDesc> tensorDescMap;
    Map<std::string, std::string> tensorDescNameMap;
    Map<std::string, std::string> tensorNameMap;
};
