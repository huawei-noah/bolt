#ifndef NORMALIZATION_OPT
#define NORMALIZATION_OPT
#include "common_opt.h"

inline EE set_normalization_opt_mali(
    bool useNchwFormat, DataType dt, char *kernelName, KernelOpt *kernelOpt)
{
    U32 len = 0;
    std::string formatName = "";
    if (useNchwFormat) {
        formatName = "_nchw";
    }

    sprintf(kernelName, "normalization%s", formatName.c_str());
    sprintf(kernelOpt->sourceName, "normalization");
    kernelOpt->kernelDataType = dt;
    char *opt = kernelOpt->option;
    if (useNchwFormat) {
        CHECK_STATUS(set_chars_define_opt("USE_NCHW", opt, &len));
        opt += len;
    }
    return SUCCESS;
}

#endif
