#ifndef ELTWISE_OPT
#define ELTWISE_OPT
#include "common_opt.h"

inline EE set_eltwise_opt_mali(U32 inputNum,
    bool useNchwFormat,
    EltwiseMode eltwiseMode,
    ActivationMode activeMode,
    DataType dt,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    U32 len = 0;
    char actName[128] = "";
    char eltName[128] = "";
    CHECK_STATUS(set_activation_modeName(activeMode, actName, &len));
    CHECK_STATUS(set_eltwise_modeName(eltwiseMode, eltName, &len));
    std::string formatName = "";
    if (useNchwFormat) {
        formatName = "nchw_";
    }

    sprintf(kernelName, "eltwise_%s%s%s%d", actName, eltName, formatName.c_str(), inputNum);
    sprintf(kernelOpt->sourceName, "eltwise");
    kernelOpt->kernelDataType = dt;
    char *opt = kernelOpt->option;
    CHECK_STATUS(set_value_define_opt(inputNum, "N", opt, &len));
    opt += len;
    CHECK_STATUS(set_activation_define_opt(activeMode, opt, &len));
    opt += len;
    CHECK_STATUS(set_eltwise_define_opt(eltwiseMode, opt, &len));
    opt += len;
    if (useNchwFormat) {
        CHECK_STATUS(set_chars_define_opt("USE_NCHW", opt, &len));
        opt += len;
    }
    return SUCCESS;
}

inline EE set_eltwise_broadcast_opt_mali(bool useNchwFormat,
    bool axisSpeMode,
    EltwiseMode eltwiseMode,
    ActivationMode activeMode,
    DataType dt,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    U32 len = 0;
    char actName[128] = "";
    char eltName[128] = "";
    CHECK_STATUS(set_activation_modeName(activeMode, actName, &len));
    CHECK_STATUS(set_eltwise_modeName(eltwiseMode, eltName, &len));
    std::string formatName = "";
    std::string axisName = "common";
    if (useNchwFormat) {
        formatName = "nchw_";
        if (axisSpeMode) {
            axisName = "axis_w1";
        }
    } else if (axisSpeMode) {
        axisName = "axis_z1";
    }

    sprintf(kernelName, "eltwise_broadcast_%s%s%s%s", actName, eltName, formatName.c_str(),
        axisName.c_str());
    sprintf(kernelOpt->sourceName, "eltwise_broadcast");
    kernelOpt->kernelDataType = dt;
    char *opt = kernelOpt->option;
    CHECK_STATUS(set_activation_define_opt(activeMode, opt, &len));
    opt += len;
    CHECK_STATUS(set_eltwise_define_opt(eltwiseMode, opt, &len));
    opt += len;
    if (useNchwFormat) {
        CHECK_STATUS(set_chars_define_opt("USE_NCHW", opt, &len));
        opt += len;
    }
    if (axisSpeMode) {
        if (useNchwFormat) {
            CHECK_STATUS(set_chars_define_opt("AXIS_W1", opt, &len));
            opt += len;
        } else {
            CHECK_STATUS(set_chars_define_opt("AXIS_Z1", opt, &len));
            opt += len;
        }
    }
    return SUCCESS;
}
#endif
