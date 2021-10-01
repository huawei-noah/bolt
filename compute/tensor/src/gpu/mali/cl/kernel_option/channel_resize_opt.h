#ifndef CHANNEL_RESIZE_OPT
#define CHANNEL_RESIZE_OPT
#include "common_opt.h"
inline EE set_channel_resize_opt_mali(bool useNchwFormat,
    DataType dt,
    GCLMemType inputMemType,
    GCLMemType outputMemType,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    std::string formatName = "";
    if (useNchwFormat) {
        formatName = "nchw_";
    }
    char ioMemName[128] = "";
    CHECK_STATUS(set_io_mem_name(inputMemType, outputMemType, ioMemName));

    sprintf(kernelName, "channel_resize_%s%s", ioMemName, formatName.c_str());
    sprintf(kernelOpt->sourceName, "channel_resize");
    kernelOpt->kernelDataType = dt;
    char *opt = kernelOpt->option;
    if (useNchwFormat) {
        CHECK_STATUS(set_chars_define_opt("USE_NCHW", opt));
    }
    CHECK_STATUS(set_io_mem_define_opt(inputMemType, outputMemType, opt));
    return SUCCESS;
}

#endif
