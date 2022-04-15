#ifndef DEPTH2SPACE_OPT
#define DEPTH2SPACE_OPT
#include "common_opt.h"

inline EE set_depth2space_nchwc4_2x2_opt(bool useOutputNchw,
    DataType dt,
    GCLMemType inputMemType,
    GCLMemType outputMemType,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    char *opt = kernelOpt->option;
    std::string outputFormatName = "";
    if (useOutputNchw) {
        outputFormatName = "_nchw";
    }
    char ioMemName[128] = "";
    CHECK_STATUS(set_io_mem_name(inputMemType, outputMemType, ioMemName));
    kernelOpt->kernelDataType = DT_F16;
    sprintf(kernelName, "depth2space_nchwc4_2x2%s", outputFormatName.c_str());
    sprintf(kernelOpt->sourceName, "depth2space_nchwc4_2x2");
    if (useOutputNchw) {
        CHECK_STATUS(set_chars_define_opt("OUT_NCHW", opt));
    }
    CHECK_STATUS(set_io_mem_define_opt(inputMemType, outputMemType, opt));
    return SUCCESS;
}
#endif
