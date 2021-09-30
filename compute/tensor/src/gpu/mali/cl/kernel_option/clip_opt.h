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
    std::string formatName = "";
    if (useNchwFormat) {
        formatName = "nchw_";
    }
    char ioMemName[128] = "";
    CHECK_STATUS(set_io_mem_name(inputMemType, outputMemType, ioMemName));
    sprintf(kernelName, "clip_%s%s", ioMemName, formatName.c_str());
    sprintf(kernelOpt->sourceName, "clip");
    if (useNchwFormat) {
        CHECK_STATUS(set_chars_define_opt("USE_NCHW", opt));
    }
    CHECK_STATUS(set_io_mem_define_opt(inputMemType, outputMemType, opt));
    return SUCCESS;
}
#endif
