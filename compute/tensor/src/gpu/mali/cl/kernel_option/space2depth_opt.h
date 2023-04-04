#ifndef SPACE2DEPTH_OPT
#define SPACE2DEPTH_OPT
#include "common_opt.h"

inline EE set_space2depth_opt(bool useFormatNchw,
    DataType dt,
    GCLMemType inputMemType,
    GCLMemType outputMemType,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    char *opt = kernelOpt->option;
    std::string name = "space2depth";
    if (useFormatNchw) {
        name += "_nchw";
    }
    char ioMemName[128] = "";
    CHECK_STATUS(set_io_mem_name(inputMemType, outputMemType, ioMemName));
    kernelOpt->kernelDataType = dt;
    UNI_STRCPY(kernelName, name.c_str());
    UNI_STRCPY(kernelOpt->sourceName, "space2depth");
    if (useFormatNchw) {
        CHECK_STATUS(set_chars_define_opt("USE_NCHW", opt));
    }
    CHECK_STATUS(set_io_mem_define_opt(inputMemType, outputMemType, opt));
    return SUCCESS;
}
#endif
