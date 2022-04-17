#ifndef NORMALIZATION_OPT
#define NORMALIZATION_OPT
#include "common_opt.h"

inline EE set_normalization_opt_mali(
    bool useNchwFormat, DataType dt, char *kernelName, KernelOpt *kernelOpt)
{
    std::string kernel = "normalization";
    if (useNchwFormat) {
        kernel += "_nchw";
    }
    UNI_STRCPY(kernelName, kernel.c_str());
    UNI_STRCPY(kernelOpt->sourceName, "normalization");
    kernelOpt->kernelDataType = dt;
    char *opt = kernelOpt->option;
    if (useNchwFormat) {
        CHECK_STATUS(set_chars_define_opt("USE_NCHW", opt));
    }
    return SUCCESS;
}

#endif
