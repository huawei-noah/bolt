#ifndef CONCAT_OPT
#define CONCAT_OPT
#include "common_opt.h"
inline EE set_concat_opt_mali(U32 concatDim,
    U32 inputNum,
    bool useNchwFormat,
    bool axisCAligned,
    DataType dt,
    char* kernelName,
    KernelOpt* kernelOpt)
{
    U32 len = 0;
    char* opt = kernelOpt->option;
    std::string formatName= "";
    if (useNchwFormat) {
        formatName = "nchw_";
    }
    std::string dimName;
    switch (concatDim) {
        case 0:
            dimName = "w_";
            CHECK_STATUS(set_chars_define_opt("AXIS_W", opt, &len));
            opt += len;
            break;
        case 1:
            dimName = "h_";
            CHECK_STATUS(set_chars_define_opt("AXIS_H", opt, &len));
            opt += len;
            break;
        case 2:
            if (!useNchwFormat && !axisCAligned) {
                dimName = "non_align_c_";
                CHECK_STATUS(set_chars_define_opt("NON_ALIGN_AXIS_C", opt, &len));
            } else {
                dimName = "c_";
                CHECK_STATUS(set_chars_define_opt("AXIS_C", opt, &len));
            }
            opt += len;
            break;
        default:
            CHECK_STATUS(NOT_SUPPORTED);
    }
    sprintf(kernelName, "concat_%s%s%d", formatName.c_str(), dimName.c_str(), inputNum);
    if (useNchwFormat) {
        sprintf(kernelOpt->sourceName, "concat_nchw");
    } else {
        sprintf(kernelOpt->sourceName, "concat");
    }
    kernelOpt->kernelDataType = dt;
    CHECK_STATUS(set_value_define_opt(inputNum, "N", opt, &len));
    return SUCCESS;
}
#endif

