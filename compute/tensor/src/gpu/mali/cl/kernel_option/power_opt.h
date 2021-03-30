#ifndef POWER_OPT
#define POWER_OPT
#include "common_opt.h"
inline EE set_power_opt_mali(bool useNchwFormat, DataType dt, char *kernelName, KernelOpt *kernelOpt)
{
    U32 len = 0;
    std::string formatName = "";
    std::string dtName = "";
    if (useNchwFormat) {
        formatName = "_nchw";
    }
    if (dt == DT_I32) {
        dtName = "_i32";
    } else if (dt != DT_F16) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    sprintf(kernelName, "power%s%s", formatName.c_str(), dtName.c_str());
    sprintf(kernelOpt->sourceName, "power");
    kernelOpt->kernelDataType = dt;
    char* opt = kernelOpt->option;
    strcpy(opt, "");
    if (useNchwFormat) {
        CHECK_STATUS(set_chars_define_opt("USE_NCHW", opt, &len));
        opt += len;
    }
    if (dt == DT_I32) {
        CHECK_STATUS(set_chars_define_opt("USE_I32", opt, &len));
        opt += len;
    }
    return SUCCESS;
}

#endif
