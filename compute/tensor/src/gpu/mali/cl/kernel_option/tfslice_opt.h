#ifndef TFSLICE_OPT
#define TFSLICE_OPT

#include "common_opt.h"

inline EE set_tfslice_opt_mali(bool useNchwFormat,
    DataType dt,
    GCLMemType inputMemType,
    GCLMemType outputMemType,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    kernelOpt->kernelDataType = dt;
    char *opt = kernelOpt->option;
    std::string formatName = "";
    if (useNchwFormat) {
        formatName = "nchw";
    }
    std::string kernel = "tfslice_" + formatName;
    UNI_STRCPY(kernelName, kernel.c_str());
    UNI_STRCPY(kernelOpt->sourceName, "tfslice");
    if (useNchwFormat) {
        CHECK_STATUS(set_chars_define_opt("USE_NCHW", opt));
    }
    return SUCCESS;
}
#endif
