#include <vector>

class Compile {
public:
    Compile(std::vector<std::string> sourceFileList, std::string compileFilePath);
    ~Compile(){}
private:
    std::string generateCompileScript();
};
