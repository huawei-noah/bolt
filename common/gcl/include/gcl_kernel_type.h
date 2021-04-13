#ifndef H_GCL_KERNEL_TYPE_H
#define H_GCL_KERNEL_TYPE_H
#include "tensor_desc.h"

struct GCLKernelSource {
    const char *data;
    const unsigned int len;
    bool use_kernel_def_head;
};

/*this can be replace by GCLKernelOptionEx after all kernel compile .sh been abolished*/
struct GCLKernelOption {
    const char *option;
    const char *sourceName;
    bool use_common_opt;
};

struct GCLKernelOptionExt {
    char option[1024];
    char sourceName[128];
    DataType kernelDataType;
};
typedef GCLKernelOptionExt KernelOpt;
#endif
