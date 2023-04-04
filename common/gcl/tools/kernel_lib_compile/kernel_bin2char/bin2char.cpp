// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <string>

int main(int argc, char *argv[])
{
    char *binName;
    char *cppName;
    char *charName;
    char *binMapName;
    std::string binFile;
    std::string cppFile;

    if (argc == 3) {
        binName = argv[1];
        binFile = binName;
        charName = strtok(binName, ".");
        cppFile = std::string(charName) + ".cpp";
        int len = strlen(charName);
        for (int i = len - 1; i > -1; --i) {
            if (charName[i] == '/') {
                charName = &charName[i + 1];
                break;
            }
        }
        binMapName = argv[2];
    } else if (argc == 4) {
        binName = argv[1];
        binFile = binName;
        cppName = argv[2];
        cppFile = cppName;
        charName = strtok(cppName, ".");
        int len = strlen(charName);
        for (int i = len - 1; i > -1; --i) {
            if (charName[i] == '/') {
                charName = &charName[i + 1];
                break;
            }
        }
        binMapName = argv[3];
    } else {
        printf("[ERROR] please pass xxx.bin name + binmapname or xxx.bin name + xxx.cpp name + "
               "binmapname.\n");
        return 1;
    }

    FILE *fpbin = fopen(binFile.c_str(), "rb");
    if (fpbin == NULL) {
        printf("[ERROR] can not open file %s.\n", binFile.c_str());
        return 1;
    }

    struct stat f_stat;
    if (stat(binFile.c_str(), &f_stat) == -1) {
        printf("[ERROR] can not get file %s size.\n", binFile.c_str());
        fclose(fpbin);
        return 1;
    }
    int filelen = f_stat.st_size;
    std::string str = "#include \"inline_" + std::string(binMapName) + ".h\"\n\nconst U32 " +
        std::string(charName) + "_len = " + std::to_string(filelen_st) + ";\nconst U8 " +
        std::string(charName) + "[] = {";
    std::stringstream ss;
    for (int i = 0; i < filelen; i++) {
        unsigned char c;
        if (i % 20 == 0) {
            ss << "\n";
        }
        if (1 != fread(&c, 1, 1, fpbin)) {
            printf("[ERROR] can not read file %s content.\n", binFile.c_str());
            fclose(fpbin);
            return 1;
        }
        ss << "0x" << std::hex << std::setw(2) << std::setfill('0') << i;
        if (i == filelen - 1) {
        } else if (i % 20 == 19) {
            ss << ",";
        } else {
            ss << ", ";
        }
    }
    str += ss.str() + "};";

    std::ofstream file;
    file.open(cppFile.c_str());
    file << str;
    file.close();

    fclose(fpbin);
    return 0;
}
