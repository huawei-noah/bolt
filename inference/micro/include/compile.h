#include <vector>

class Compile {
public:
    Compile(std::vector<std::string> sourceFileList, std::string compileFilePath);

private:
    std::string generateCompileScript();
};
