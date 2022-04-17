#ifndef ROIALIGN_OPT
#define ROIALIGN_OPT
#include "common_opt.h"
inline EE set_roialign_opt_mali(bool useNchwFormat,
    PoolingMode mode,
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
        formatName = "nchw_";
    }
    char ioMemName[128] = "";
    CHECK_STATUS(set_io_mem_name(inputMemType, outputMemType, ioMemName));
    std::string modeName = "";
    if (mode == POOLING_MAX) {
        modeName = "max";
        CHECK_STATUS(set_chars_define_opt("USE_MAX", opt));
    } else if (mode == POOLING_MEAN) {
        modeName = "avg";
        CHECK_STATUS(set_chars_define_opt("USE_AVG", opt));
    } else {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    std::string kernel = std::string("roialign_") + ioMemName + formatName + modeName;
    UNI_STRCPY(kernelName, kernel.c_str());
    UNI_STRCPY(kernelOpt->sourceName, "roialign");
    if (useNchwFormat) {
        CHECK_STATUS(set_chars_define_opt("USE_NCHW", opt));
    }
    CHECK_STATUS(set_io_mem_define_opt(inputMemType, outputMemType, opt));
    return SUCCESS;
}
#endif
