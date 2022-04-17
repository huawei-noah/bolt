#ifndef CLIP_OPT
#define CLIP_OPT
#include "common_opt.h"
inline EE set_clip_opt_mali(bool useNchwFormat,
    DataType dt,
    GCLMemType inputMemType,
    GCLMemType outputMemType,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    char *opt = kernelOpt->option;
    kernelOpt->kernelDataType = dt;
    char ioMemName[128] = "";
    CHECK_STATUS(set_io_mem_name(inputMemType, outputMemType, ioMemName));
    std::string name = "clip_" + std::string(ioMemName);
    if (useNchwFormat) {
        name += "nchw_";
    }
    UNI_STRCPY(kernelName, name.c_str());
    UNI_STRCPY(kernelOpt->sourceName, "clip");
    if (useNchwFormat) {
        CHECK_STATUS(set_chars_define_opt("USE_NCHW", opt));
    }
    CHECK_STATUS(set_io_mem_define_opt(inputMemType, outputMemType, opt));
    return SUCCESS;
}
#endif
