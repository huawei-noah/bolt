#ifndef SLICE_OPT
#define SLICE_OPT
#include "common_opt.h"
inline EE set_slice_opt_mali(
    bool useNchwFormat, U32 axis, U32 slice_num, DataType dt, char *kernelName, KernelOpt *kernelOpt)
{
    std::string formatName = "";
    if (useNchwFormat) {
        formatName = "nchw_";
    }

    sprintf(kernelName, "slice_%s%d%d", formatName.c_str(), axis, slice_num);
    sprintf(kernelOpt->sourceName, "slice");
    kernelOpt->kernelDataType = dt;
    char *opt = kernelOpt->option;
    CHECK_STATUS(set_value_define_opt(axis, "AXIS_NUM", opt));
    CHECK_STATUS(set_value_define_opt(slice_num, "ON", opt));
    if (useNchwFormat) {
        CHECK_STATUS(set_chars_define_opt("USE_NCHW", opt));
    }
    return SUCCESS;
}

#endif
