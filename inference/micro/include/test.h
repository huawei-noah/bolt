#include <string>

class Test {
public:
    Test(std::string testFilePath);

    void generate()
    {
        std::string code = "";
        code += generateTestHeader();
        code += generateTest();
        writeTestToFile(code, testFilePath);
    }

private:
    std::string generateTestHeader();

    std::string generateTest();

    std::string testFilePath;
};
