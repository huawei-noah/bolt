#ifndef SLICE_OPT
#define SLICE_OPT
#include "common_opt.h"
inline EE set_slice_opt_mali(
    bool useNchwFormat, U32 axis, U32 slice_num, DataType dt, char *kernelName, KernelOpt *kernelOpt)
{
    std::string name = "slice_";
    if (useNchwFormat) {
        name += "nchw_";
    }
    name += std::to_string(axis) + std::to_string(slice_num);
    UNI_STRCPY(kernelName, name.c_str());
    UNI_STRCPY(kernelOpt->sourceName, "slice");
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
