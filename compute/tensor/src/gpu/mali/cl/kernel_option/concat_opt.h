#ifndef CONCAT_OPT
#define CONCAT_OPT
#include "common_opt.h"
inline EE set_concat_opt_mali(U32 concatDim,
    U32 inputNum,
    bool useNchwFormat,
    bool axisAligned,
    DataType dt,
    GCLMemType *inputMemType,
    GCLMemType outputMemType,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    char *opt = kernelOpt->option;
    std::string formatName = "";
    if (useNchwFormat) {
        formatName = "nchw_";
    }
    std::string dimName;
    switch (concatDim) {
        case 0:
            if (useNchwFormat && !axisAligned) {
                dimName = "non_align_w_";
                CHECK_STATUS(set_chars_define_opt("NON_ALIGN_AXIS_W", opt));
            } else {
                dimName = "w_";
                CHECK_STATUS(set_chars_define_opt("AXIS_W", opt));
            }
            break;
        case 1:
            dimName = "h_";
            CHECK_STATUS(set_chars_define_opt("AXIS_H", opt));
            break;
        case 2:
            if (!useNchwFormat && !axisAligned) {
                dimName = "non_align_c_";
                CHECK_STATUS(set_chars_define_opt("NON_ALIGN_AXIS_C", opt));
            } else {
                dimName = "c_";
                CHECK_STATUS(set_chars_define_opt("AXIS_C", opt));
            }
            break;
        default:
            CHECK_STATUS(NOT_SUPPORTED);
    }
    char iomName[128] = "";
    CHECK_STATUS(
        set_io_mems_name_and_define_opts(inputMemType, &outputMemType, inputNum, 1, iomName, opt));
    std::string kernel = "concat_" + formatName + iomName + dimName + std::to_string(inputNum);
    UNI_STRCPY(kernelName, kernel.c_str());
    if (useNchwFormat) {
        UNI_STRCPY(kernelOpt->sourceName, "concat_nchw");
    } else {
        UNI_STRCPY(kernelOpt->sourceName, "concat");
    }
    kernelOpt->kernelDataType = dt;
    CHECK_STATUS(set_value_define_opt(inputNum, "N", opt));
    return SUCCESS;
}
#endif
