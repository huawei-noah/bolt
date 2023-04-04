// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gcl.h"
#include <getopt.h>

const char *imagesource = "#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable\n";

const char *half16source = "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";

void printHelp()
{
    printf("please use the linux tradition, or you will face problem!!!!!!!!!!!!!!\n");
    printf("The program only support opencl kernel compile now !!!!!!!!!!!1!!!!!!!\n");
    printf("-i or --input to specify OpenCL input cl soruce file name\n");
    printf("-o or --output to specify OpenCL output binary file name\n");
    printf("-O or --options to specify OpenCL compiling options\n");
}

bool GetFileLength(const char *filename, U32 *len)
{
    if ((NULL == filename) || (0 == strlen(filename))) {
        return false;
    }
    FILE *fp = fopen(filename, "rb");
    if (NULL == fp) {
        return false;
    }
    rewind(fp);
    if (0 != fseek(fp, 0, SEEK_END)) {
        return false;
    }
    *len = ftell(fp);
    fclose(fp);
    return true;
}

bool LoadBinFile(const char *filename, I8 *str, U32 len)
{
    if ((NULL == filename) || (0 == strlen(filename))) {
        return false;
    }
    FILE *fp = fopen(filename, "rb");
    if (NULL == fp) {
        return false;
    }
    rewind(fp);
    if (len != fread(str, sizeof(char), len, fp)) {
        fclose(fp);
        return false;
    }
    fclose(fp);
    return true;
}

bool StoreToBinFile(const char *filename, U32 length, const U8 *s)
{
    if ((NULL == s) || (NULL == filename) || (0 == strlen(filename))) {
        return false;
    }
    FILE *fp = fopen(filename, "wb");
    if (NULL == fp) {
        return false;
    }
    if (length != fwrite(s, sizeof(char), length, fp)) {
        fclose(fp);
        return false;
    }
    fclose(fp);
    return true;
}

int parseCommandLine(I32 argc, I8 *argv[], I8 **inputFilename, I8 **outputFilename, I8 **options)
{
    const struct option long_options[] = {{"input", 1, nullptr, 'i'}, {"output", 1, nullptr, 'o'},
        {"options", 1, nullptr, 'O'}, {nullptr, 1, nullptr, '0'}};
    bool setInput = false;
    bool setOutput = false;
    bool setOptions = false;
    int optionIndex = 0;
    int ch;
    while ((ch = getopt_long(argc, argv, "i:o:O", long_options, &optionIndex)) != -1) {
        switch (ch) {
            case 'i':
                printf("input file name is %s\n", optarg);
                *inputFilename = optarg;
                if (setInput) {
                    printf("you specify input file name twice, program will exit\n");
                    return 1;
                }
                setInput = true;
                break;
            case 'o':
                printf("output file name is %s\n", optarg);
                *outputFilename = optarg;
                if (setOutput) {
                    printf("you specify output file name twice, program will exit\n");
                    return 1;
                }
                setOutput = true;
                break;
            case 'O':
                printf("options is %s\n", optarg);
                *options = optarg;
                if (setOptions) {
                    printf("you specify compiling options twice, program will exit\n");
                    return 1;
                }
                setOptions = true;
                break;
            default:
                printf("not support option:%c\n", ch);
                return 1;
        }
    }
    if (!setInput) {
        printf("you don't specify the input cl file name, program will exit\n");
        return 1;
    }
    if (!setOutput) {
        printf("you don't specify the output file name, program will exit\n");
        return 1;
    }
    if (!setOptions) {
        printf("you don't specify the options for compiling cl file, default is empty\n");
        *options = (char *)"";
    }
    return 0;
}

int main(I32 argc, I8 *argv[])
{
    if (1 == argc) {
        printHelp();
        return 1;
    }

    I8 *FLAGS_inputFilename;
    I8 *FLAGS_outputFilename;
    I8 *FLAGS_options;
    if (0 !=
        parseCommandLine(argc, argv, &FLAGS_inputFilename, &FLAGS_outputFilename, &FLAGS_options)) {
        return 1;
    }

    GCLHandle_t handle;
    CHECK_STATUS(gcl_create_handle(&handle));
    U32 imageLen = 0;
#ifdef CL_VERSION_1_2
    imageLen = strlen(imagesource);
#endif
    U32 half16Len = strlen(half16source);
    U32 clcodeLen = 0;
    bool FileStatus = GetFileLength(FLAGS_inputFilename, &clcodeLen);
    if (!FileStatus) {
        printf("get file length failed\n");
        return 1;
    }
    U32 srcLen = imageLen + half16Len + clcodeLen;
    I8 *source = new I8[srcLen];
#ifdef CL_VERSION_1_2
    UNI_MEMCPY(source, imagesource, imageLen);
#endif
    UNI_MEMCPY(source + imageLen, half16source, half16Len);
    FileStatus = LoadBinFile(FLAGS_inputFilename, source + imageLen + half16Len, clcodeLen);
    if (!FileStatus) {
        printf("load bin file failed\n");
        delete[] source;
        return 1;
    }

    Program program;
    U32 numKernel = 1;
    Kernel kernel;
    U32 size = 0;
    U8 *binary;

    CHECK_STATUS(gcl_produce_program_kernel_with_source(
        handle, &srcLen, source, FLAGS_options, &program, numKernel, &kernel));
    CHECK_STATUS(gcl_get_program_info(program, &binary, &size));
    FileStatus = StoreToBinFile(FLAGS_outputFilename, size, binary);
    if (!FileStatus) {
        printf("store bin file failed\n");
        return 1;
    }
    free(binary);
    delete[] source;
    CHECK_STATUS(release_program(program));
    CHECK_STATUS(release_kernel(kernel));
    gcl_destroy_handle(handle);
    return 0
}
