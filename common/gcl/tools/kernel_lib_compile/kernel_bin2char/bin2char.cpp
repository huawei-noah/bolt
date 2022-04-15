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
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <string>
#include <string.h>

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
        printf("please input .bin name + binmapname or input .bin name + .cpp name + binmapname\n");
    }

    FILE *fpbin = fopen(binFile.c_str(), "rb");
    if (fpbin == NULL) {
        printf("file %s open error\n", binFile.c_str());
        return 1;
    }

    struct stat f_stat;
    if (stat(binFile.c_str(), &f_stat) == -1) {
        printf("file %s get size error\n", binFile.c_str());
        fclose(fpbin);
        return 1;
    }
    int filelen = f_stat.st_size;
    std::stringstream templen;
    templen << filelen;
    std::string filelen_st = templen.str();

    std::string str = "#include \"inline_" + std::string(binMapName) + ".h\"\n\nCU32 " +
        std::string(charName) + "_len = " + filelen_st + ";\nCU8 " + std::string(charName) +
        "[] = {";

    unsigned char charRead;
    std::string appendBuf;

    for (int i = 0; i < filelen; i++) {
        appendBuf.clear();
        if (i % 20 == 0) {
            appendBuf += "\n";
        }
        if (1 != fread(&charRead, 1, 1, fpbin)) {
            printf("file %s read error\n", binFile.c_str());
            fclose(fpbin);
            return 1;
        }
        char tempstr[4];
        sprintf(tempstr, "0x%02x", charRead);
        appendBuf += std::string(tempstr);

        if (i == filelen - 1) {
        } else if (i % 20 == 19) {
            appendBuf += ",";
        } else {
            appendBuf += ", ";
        }
        str += appendBuf;
    }

    str += "};";

    std::ofstream file;
    file.open(cppFile.c_str());
    file << str;
    file.close();

    fclose(fpbin);

    return 0;
}
