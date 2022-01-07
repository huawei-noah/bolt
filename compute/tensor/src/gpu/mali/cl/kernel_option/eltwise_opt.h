#ifndef ELTWISE_OPT
#define ELTWISE_OPT
#include "common_opt.h"

inline EE set_eltwise_opt_mali(U32 inputNum,
    bool useNchwFormat,
    EltwiseMode eltwiseMode,
    ActivationMode activeMode,
    DataType dt,
    GCLMemType *inputMemType,
    GCLMemType outputMemType,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    char *opt = kernelOpt->option;
    char actName[128] = "";
    char eltName[128] = "";
    CHECK_STATUS(set_activation_mode_name(activeMode, actName));
    CHECK_STATUS(set_eltwise_mode_name(eltwiseMode, eltName));
    std::string formatName = "";
    if (useNchwFormat) {
        formatName = "nchw_";
    }
    char iomName[128] = "";
    CHECK_STATUS(
        set_io_mems_name_and_define_opts(inputMemType, &outputMemType, inputNum, 1, iomName, opt));
    sprintf(
        kernelName, "eltwise_%s%s%s%s%d", iomName, actName, eltName, formatName.c_str(), inputNum);
    sprintf(kernelOpt->sourceName, "eltwise");
    kernelOpt->kernelDataType = dt;
    CHECK_STATUS(set_value_define_opt(inputNum, "N", opt));
    CHECK_STATUS(set_activation_define_opt(activeMode, opt));
    CHECK_STATUS(set_eltwise_define_opt(eltwiseMode, opt));
    if (useNchwFormat) {
        CHECK_STATUS(set_chars_define_opt("USE_NCHW", opt));
    }
    return SUCCESS;
}

inline EE set_eltwise_broadcast_opt_mali(bool useNchwFormat,
    bool axisSpeMode,
    U32 arrayDimMax,
    EltwiseMode eltwiseMode,
    ActivationMode activeMode,
    DataType dt,
    GCLMemType *inputMemType,
    GCLMemType outputMemType,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    kernelOpt->kernelDataType = dt;
    char *opt = kernelOpt->option;
    char actName[128] = "";
    char eltName[128] = "";
    CHECK_STATUS(set_activation_mode_name(activeMode, actName));
    CHECK_STATUS(set_eltwise_mode_name(eltwiseMode, eltName));
    std::string formatName = "";
    std::string axisName = "common";
    if (useNchwFormat) {
        formatName = "nchw_";
        if (axisSpeMode) {
            axisName = "axis_w1";
        }
    } else if (axisSpeMode) {
        axisName = "axis_c1";
    }
    std::string swapInputName = "";
    if (arrayDimMax == 1) {
        if (eltwiseMode == ELTWISE_SUB || eltwiseMode == ELTWISE_DIV) {
            swapInputName = "si_";
            CHECK_STATUS(set_chars_define_opt("SWAP_INPUT", opt));
        }
    }
    char iomName[128] = "";
    CHECK_STATUS(set_io_mems_name_and_define_opts(inputMemType, &outputMemType, 2, 1, iomName, opt));

    sprintf(kernelName, "eltwise_broadcast_%s%s%s%s%s%s", iomName, actName, eltName,
        swapInputName.c_str(), formatName.c_str(), axisName.c_str());
    sprintf(kernelOpt->sourceName, "eltwise_broadcast");
    CHECK_STATUS(set_activation_define_opt(activeMode, opt));
    CHECK_STATUS(set_eltwise_define_opt(eltwiseMode, opt));
    if (useNchwFormat) {
        CHECK_STATUS(set_chars_define_opt("USE_NCHW", opt));
    }
    if (axisSpeMode) {
        if (useNchwFormat) {
            CHECK_STATUS(set_chars_define_opt("AXIS_W1", opt));
        } else {
            CHECK_STATUS(set_chars_define_opt("AXIS_C1", opt));
        }
    }
    return SUCCESS;
}
#endif
