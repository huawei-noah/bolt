#include <string>

class Util {
public:
    Util(std::string utilHeaderFilePath)
    {}

    void generate()
    {
        std::string code = "";
        code += generateUtilFileHeader;
        code += generateStruct;

        code += generateLogFunction;
        writeUtilToFile(code, this->utilHeaderFilePath);
    }

private:
    void generateUtilHeader(std::string utilHeaderFilePath);

    // generate iot code log function
    std::string generateLogFunction();

    // generate iot code struct
    std::string generateStruct();

    std::string utilHeaderFilePath;
};
