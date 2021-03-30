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
#include <vector>
#include <map>
#include <string>
#include <dirent.h>

#include "data_type.h"
#include "error.h"

typedef struct {
    std::string kernel;
    U32 len;
    bool use_kernel_def_head;
} KernelInfo;

typedef struct {
    std::string sourceName;
    std::string option;
    bool use_common_opt;
} OptionInfo;

inline std::vector<std::string> buildFileNames(std::string path, std::string postfix)
{
    struct dirent *dirTp;
    DIR *handle = opendir(path.c_str());
    std::vector<std::string> names;
    if (handle != NULL) {
        while ((dirTp = readdir(handle)) != NULL) {
            std::string clFileName = dirTp->d_name;
            U32 len = clFileName.size();
            U32 postfix_len = postfix.size();
            if (len > postfix_len) {
                if (clFileName.substr(len - postfix_len) == postfix) {
                    clFileName.erase(len - postfix_len, postfix_len);
                    names.push_back(clFileName);
                }
            }
        }
    } else {
        UNI_ERROR_LOG("opendir %s failed\n", path.c_str());
    }
    closedir(handle);
    return names;
}

inline std::string readBinaryFile(std::string fileName)
{
    std::string fileContent, line;
    std::ifstream file;
    file.open(fileName);
    if (file.is_open()) {
        while (getline(file, line)) {
            fileContent += line + "\n";
        }
        file.close();
    } else {
        UNI_ERROR_LOG("Write file %s failed.\n", fileName.c_str());
    }
    return fileContent;
}

inline void writeTextFile(std::string fileName, std::string fileContent)
{
    std::ofstream file(fileName.c_str());
    if (file.is_open()) {
        file << fileContent.c_str();
        file.close();
    } else {
        UNI_ERROR_LOG("Write file %s failed.\n", fileName.c_str());
    }
}

inline std::map<std::string, KernelInfo> buildClMap(std::vector<std::string> clNames,
    std::vector<std::string> clPaths,
    std::vector<U32> clNamesIndex,
    std::string postfix)
{
    std::map<std::string, KernelInfo> clMap;
    for (int ii = 0; ii < clPaths.size(); ii++) {
        std::string clPath = clPaths[ii];
        int be = (ii - 1) < 0 ? 0 : clNamesIndex[ii - 1];
        int end = clNamesIndex[ii];
        for (int i = be; i < end; i++) {
            KernelInfo kernelInfo;
            std::string clName = clNames[i];
            std::string fileName = clPath + clName + postfix;
            std::string fileContent = readBinaryFile(fileName);
            int note_pos = -1;
            int j = 0;
            for (; j < fileContent.size() - 1; j++) {
                if (fileContent[j] == '/' && note_pos < 0) {
                    if (fileContent[j + 1] == '/') {
                        note_pos = j;
                        continue;
                    }
                }

                if (fileContent[j] == '\n' && note_pos >= 0) {
                    fileContent.erase(note_pos, j - note_pos);
                    j = note_pos;
                    note_pos = -1;
                }
            }
            note_pos = -1;
            for (j = 0; j < fileContent.size() - 1; j++) {
                if (fileContent[j] == '/' && note_pos < 0) {
                    if (fileContent[j + 1] == '*') {
                        note_pos = j;
                        continue;
                    }
                }

                if (fileContent[j] == '*' && note_pos >= 0) {
                    if (fileContent[j + 1] == '/') {
                        fileContent.erase(note_pos, j - note_pos + 2);
                        j = note_pos;
                        note_pos = -1;
                    }
                }
            }

            for (j = 0; j < fileContent.size() - 1; j++) {
                if (fileContent[j] == '\r') {
                    fileContent.erase(j, 1);
                    j = j - 1;
                }
            }

            for (j = 0; j < fileContent.size() - 1; j++) {
                if (fileContent[j] == '\n') {
                    if (fileContent[j + 1] == '\n') {
                        fileContent.erase(j, 1);
                        j = j - 1;
                    }
                }
            }
            if (fileContent[0] == '\n') {
                fileContent.erase(0, 1);
            }
            if (fileContent[fileContent.size() - 1] == '\n') {
                fileContent.erase(fileContent.size() - 1, 1);
            }
            kernelInfo.len = fileContent.size();

            std::string kernel_def_head = "kernel_def.h";
            kernelInfo.use_kernel_def_head = false;
            if (fileContent.find(kernel_def_head, 0) != -1) {
                kernelInfo.use_kernel_def_head = true;
            }

            std::string substr_a = "\\";
            std::string substr_b = "\\n\"";
            std::string substr_c = "\"";
            U32 sublen_a = substr_a.size();
            U32 sublen_b = substr_b.size();
            U32 sublen_c = substr_c.size();

            for (j = 0; j < fileContent.size() - 1; j++) {
                if (fileContent[j] == '\\') {
                    fileContent.insert(j, substr_a);
                    j += sublen_a + 1;
                }
            }

            for (j = 0; j < fileContent.size() - 1; j++) {
                if (fileContent[j] == '"') {
                    fileContent.insert(j, substr_a);
                    j += sublen_a + 1;
                }
            }

            for (j = 0; j < fileContent.size() - 1; j++) {
                if (fileContent[j] == '\n') {
                    fileContent.insert(j, substr_b);
                    j += sublen_b + 1;
                    fileContent.insert(j, substr_c);
                    j += sublen_c;
                }
            }
            fileContent.insert(0, substr_c);
            fileContent.insert(fileContent.size(), substr_b);
            kernelInfo.kernel = fileContent;
            clMap[clName] = kernelInfo;
        }
    }
    return clMap;
}

inline std::map<std::string, std::string> buildClOptionMap(
    std::vector<std::string> optionNames, std::string optionPath, std::string postfix)
{
    std::map<std::string, std::string> optionMap;
    for (U32 i = 0; i < optionNames.size(); i++) {
        std::string optionName = optionNames[i];
        std::string fileName = optionPath + optionName + postfix;
        std::string fileContent = readBinaryFile(fileName);
        int note_pos = -1;
        int j = 0;
        for (; j < fileContent.size() - 1; j++) {
            if (fileContent[j] == '#' && note_pos < 0) {
                note_pos = j;
                continue;
            }

            if (fileContent[j] == '\n' && note_pos >= 0) {
                fileContent.erase(note_pos, j - note_pos);
                j = note_pos;
                note_pos = -1;
            }
        }

        for (j = 0; j < fileContent.size() - 1; j++) {
            if (fileContent[j] == '\r') {
                fileContent.erase(j, 1);
                j = j - 1;
            }
        }

        for (j = 0; j < fileContent.size() - 1; j++) {
            if (fileContent[j] == '\n') {
                if (fileContent[j + 1] == '\n') {
                    fileContent.erase(j, 1);
                    j = j - 1;
                }
            }
        }
        if (fileContent[0] == '\n') {
            fileContent.erase(0, 1);
        }
        if (fileContent[fileContent.size() - 1] == '\n') {
            fileContent.erase(fileContent.size() - 1, 1);
        }
        optionMap[optionName] = fileContent;
    }
    return optionMap;
}

inline std::map<std::string, OptionInfo> buildClOptionExpandMap(
    std::map<std::string, std::string> optionMap)
{
    std::map<std::string, OptionInfo> optionMapExpand;
    std::string output_flag = "--output=";
    std::string option_flag = "\\\"";
    std::string postfix = ".bin";
    std::string replace_name = "${file%.*}";
    std::string common_opt = "${copt}";
    for (auto p : optionMap) {
        std::string name = p.first;
        std::string option = p.second;
        OptionInfo optionInfo;
        optionInfo.sourceName = name;
        int pos = option.find(output_flag, 0);
        while (pos != -1) {
            int be = pos + output_flag.size();
            int end = option.find(" ", be);
            std::string expandName = option.substr(be, end - be);
            expandName.erase(expandName.size() - postfix.size(), postfix.size());
            expandName.replace(0, replace_name.size(), name);

            pos = option.find(option_flag, end);
            be = pos + option_flag.size();
            end = option.find(option_flag, be);
            std::string expandOption = option.substr(be, end - be);
            int common_opt_pos = expandOption.find(common_opt, 0);
            if (common_opt_pos == -1) {
                optionInfo.use_common_opt = false;
            } else {
                optionInfo.use_common_opt = true;
                if (name == "common") {
                    expandOption.replace(0, common_opt.size(),
                        "-cl-std=CL2.0 -D T=half -D T2=half2 -D T3=half3 -D T4=half4 -D T8=half8 "
                        "-D T16=half16 "
                        "-DUSE_HALF");
                } else {
                    expandOption.erase(common_opt_pos, common_opt.size());
                }
            }
            pos = option.find(output_flag, end);
            optionInfo.option = expandOption;
            optionMapExpand[expandName] = optionInfo;
        }
    }
    return optionMapExpand;
}

inline std::string produce_inline_cl_source_head(std::vector<std::string> clNames)
{
    std::string source_head = "";
    for (auto p : clNames) {
        std::string func = "source_" + p;
        source_head += "extern bool " + func + "_head;\n";
        source_head += "extern const unsigned int " + func + "_len;\n";
        source_head += "extern const char " + func + "[];\n";
    }
    return source_head;
}

inline std::string produce_inline_cl_option_head(std::vector<std::string> optionNamesExpand)
{
    std::string option_head = "";
    for (auto p : optionNamesExpand) {
        std::string func = "option_" + p;
        option_head += "extern bool " + func + "_common;\n";
        option_head += "extern const char " + func + "_source_name[];\n";
        option_head += "extern const char " + func + "[];\n";
    }
    return option_head;
}

inline std::string produce_inline_cl_source(std::vector<std::string> clNames)
{
    std::string source = "";
    for (auto p : clNames) {
        std::string func = "source_" + p;
        source += "    put_source(\"" + p + "\", " + "{" + func + ", " + func + "_len, " + func +
            "_head});\n";
    }
    return source;
}

inline std::string produce_inline_cl_option(std::vector<std::string> optionNamesExpand)
{
    std::string source = "";
    for (auto p : optionNamesExpand) {
        std::string func = "option_" + p;
        source += "    put_option(\"" + p + "\", " + "{" + func + ", " + func + "_source_name, " +
            func + "_common});\n";
    }
    return source;
}

inline std::string produce_kernel_source(std::string name, KernelInfo kernelInfo)
{
    name = "source_" + name;
    std::string source = "";
    bool use_kernel_def_head = kernelInfo.use_kernel_def_head;
    U32 len = kernelInfo.len;
    source += "bool " + name + "_head = " + std::to_string(use_kernel_def_head) + ";\n";
    source += "const unsigned int " + name + "_len = " + std::to_string(len) + ";\n";
    source += "const char " + name + "[] = \n";
    source += kernelInfo.kernel;
    source += ";\n";
    return source;
}

inline std::string produce_option_source(std::string name, OptionInfo optionInfo)
{
    name = "option_" + name;
    std::string source = "";
    source += "bool " + name + "_common = " + std::to_string(optionInfo.use_common_opt) + ";\n";
    source += "const char " + name + "_source_name[] = ";
    source += "\"";
    source += optionInfo.sourceName;
    source += "\";\n";
    source += "const char " + name + "[] = ";
    source += "\"";
    source += optionInfo.option;
    source += "\";\n";
    return source;
}

int main()
{
    CI8 *boltEnv = getenv("BOLT_ROOT");
    if (boltEnv == NULL) {
        UNI_ERROR_LOG("BOLT_ROOT env value has not been set successfully\n");
    };
    std::string boltPath = boltEnv;
    CI8 lastFlag = boltPath[boltPath.length() - 1];
    if (strcmp(&lastFlag, "/") != 0) {
        boltPath += "/";
    }
    std::string tensorComputingClPath = "compute/tensor/src/gpu/mali/cl/";
    std::string imageClPath = "compute/image/src/gpu/mali/cl/";
    tensorComputingClPath = boltPath + tensorComputingClPath;
    imageClPath = boltPath + imageClPath;

    std::string clOptionPath = "common/gcl/tools/kernel_lib_compile/sh/compile/";
    clOptionPath = boltPath + clOptionPath;

    std::vector<std::string> clPath;
    clPath.push_back(tensorComputingClPath);
    clPath.push_back(imageClPath);

    std::vector<std::string> clNames;
    std::vector<std::string> headNames;
    std::vector<U32> clNamesIndex;
    std::vector<U32> headNamesIndex;

    for (auto p : clPath) {
        std::vector<std::string> clName;
        std::vector<std::string> headName;
        headName = buildFileNames(p, ".h");
        clName = buildFileNames(p, ".cl");
        clNames.insert(clNames.end(), clName.begin(), clName.end());
        headNames.insert(headNames.end(), headName.begin(), headName.end());
        clNamesIndex.push_back(clNames.size());
        headNamesIndex.push_back(headName.size());
    }

    std::vector<std::string> clOptionNames;
    std::vector<std::string> clOptionNamesExpand;
    clOptionNames = buildFileNames(clOptionPath, ".sh");

    std::map<std::string, KernelInfo> headMap;
    std::map<std::string, KernelInfo> clMap;
    std::map<std::string, std::string> clOptionMap;
    std::map<std::string, OptionInfo> clOptionMapExpand;
    headMap = buildClMap(headNames, clPath, headNamesIndex, ".h");
    clMap = buildClMap(clNames, clPath, clNamesIndex, ".cl");
    clOptionMap = buildClOptionMap(clOptionNames, clOptionPath, ".sh");

    std::string filePath = "common/gcl/tools/kernel_source_compile/include/";
    filePath = boltPath + filePath;
    std::string kernel_source_executor;
    kernel_source_executor = "#ifndef _LIBKERNELSOURCE_H\n";
    kernel_source_executor += "#define _LIBKERNELSOURCE_H\n";
    kernel_source_executor += "#include \"gcl_kernel_source.h\"\n";
    kernel_source_executor += "class kernel_source_executor : public gcl_kernel_source {\n";
    kernel_source_executor += "public:\n";
    kernel_source_executor += "    kernel_source_executor() {\n";
    kernel_source_executor += "        loadKernelSource();\n";
    kernel_source_executor += "        loadKernelOption();\n";
    kernel_source_executor += "    }\n";
    kernel_source_executor += "    void loadKernelSource();\n";
    kernel_source_executor += "    void loadKernelOption();\n";
    kernel_source_executor += "};\n";
    kernel_source_executor += "#endif\n";
    writeTextFile(filePath + "libkernelsource.h", kernel_source_executor);

    filePath = "common/gcl/tools/kernel_source_compile/src/cl/";
    filePath = boltPath + filePath;
    std::string kernel_source = "#include \"libkernelsource.h\"\n";
    for (auto p : headMap) {
        std::string name = p.first;
        KernelInfo kernelInfo = p.second;
        kernel_source += produce_kernel_source(name, kernelInfo);
    }
    for (auto p : clMap) {
        std::string name = p.first;
        KernelInfo kernelInfo = p.second;
        kernel_source += produce_kernel_source(name, kernelInfo);
    }
    kernel_source += "void kernel_source_executor::loadKernelSource() {\n";
    kernel_source += produce_inline_cl_source(headNames);
    kernel_source += produce_inline_cl_source(clNames);
    kernel_source += "}\n";
    writeTextFile(filePath + "gcl_kernel_source.cpp", kernel_source);

    clOptionMapExpand = buildClOptionExpandMap(clOptionMap);
    for (auto p : clOptionMapExpand) {
        clOptionNamesExpand.push_back(p.first);
    }
    filePath = "common/gcl/tools/kernel_source_compile/src/option/";
    filePath = boltPath + filePath;

    std::string option_source = "#include \"libkernelsource.h\"\n";
    for (auto p : clOptionMapExpand) {
        std::string name = p.first;
        OptionInfo optionInfo = p.second;
        option_source += produce_option_source(name, optionInfo);
    }
    option_source += "void kernel_source_executor::loadKernelOption() {\n";
    option_source += produce_inline_cl_option(clOptionNamesExpand);
    option_source += "}\n";
    writeTextFile(filePath + "gcl_kernel_option.cpp", option_source);
    return 0;
}
