#ifndef POWER_OPT
#define POWER_OPT
#include "common_opt.h"
inline EE set_power_opt_mali(bool useNchwFormat,
    DataType dt,
    GCLMemType inputMemType,
    GCLMemType outputMemType,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    std::string formatName = "";
    std::string dtName = "";
    if (useNchwFormat) {
        formatName = "nchw";
    }
    if (dt == DT_I32) {
        dtName = "_i32";
    } else if (dt != DT_F16) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    char ioMemName[128] = "";
    CHECK_STATUS(set_io_mem_name(inputMemType, outputMemType, ioMemName));
    sprintf(kernelName, "power_%s%s%s", ioMemName, formatName.c_str(), dtName.c_str());
    sprintf(kernelOpt->sourceName, "power");
    kernelOpt->kernelDataType = dt;
    char *opt = kernelOpt->option;
    if (useNchwFormat) {
        CHECK_STATUS(set_chars_define_opt("USE_NCHW", opt));
    }
    if (dt == DT_I32) {
        CHECK_STATUS(set_chars_define_opt("USE_I32", opt));
    }
    CHECK_STATUS(set_io_mem_define_opt(inputMemType, outputMemType, opt));
    return SUCCESS;
}

#endif
