#ifndef CONV_DEPTHWISE_OPT
#define CONV_DEPTHWISE_OPT
#include "common_opt.h"
inline EE set_conv_depthwise_opt_mali(U32 fw,
    U32 fh,
    U32 sw,
    U32 workEntriesPerThread,
    ActivationMode activeMode,
    bool outputNcwhMode,
    DataType dt,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    U32 len = 0;
    U32 ON = workEntriesPerThread;
    char modeName[128] = "";
    CHECK_STATUS(set_activation_modeName(activeMode, modeName, &len));
    std::string formatName = "";
    if (outputNcwhMode) {
        formatName = "ncwh_";
    }
    sprintf(
        kernelName, "conv_depthwise_sw%d_%s%s%d%d%d", sw, modeName, formatName.c_str(), fw, fh, ON);

    sprintf(kernelOpt->sourceName, "conv_depthwise_sw%d", sw);
    kernelOpt->kernelDataType = dt;
    char *opt = kernelOpt->option;
    if (ON < 1 || ON > 8) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    U32 IN, LN, UN, FWH;
    bool useBasicReg = true;
    IN = ON;
    LN = IN - 1;
    UN = LN;
    FWH = fw * fh;
    if (sw == 1) {
        if (fw < 3) {
            useBasicReg = false;
        } else if (fw == 3 && ON <= 5) {
            useBasicReg = false;
        } else if (fw <= 5 && ON <= 3) {
            useBasicReg = false;
        } else if (fw <= 7 && ON <= 2) {
            useBasicReg = false;
        }
    } else if (sw == 2) {
        if (fw <= 3 && ON <= 3) {
            useBasicReg = false;
        }
    } else {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    if (!useBasicReg) {
        IN = (ON - 1) * sw + fw;
        LN = IN;
        UN = LN - 1;
    }

    CHECK_STATUS(set_value_define_opt(fw, "FW", opt, &len));
    opt += len;
    CHECK_STATUS(set_value_define_opt(fh, "FH", opt, &len));
    opt += len;
    CHECK_STATUS(set_value_define_opt(ON, "ON", opt, &len));
    opt += len;
    CHECK_STATUS(set_value_define_opt(IN, "IN", opt, &len));
    opt += len;
    CHECK_STATUS(set_value_define_opt(LN, "LN", opt, &len));
    opt += len;
    CHECK_STATUS(set_value_define_opt(LN, "UN", opt, &len));
    opt += len;
    CHECK_STATUS(set_value_define_opt(FWH, "FWH", opt, &len));
    opt += len;
    CHECK_STATUS(set_activation_define_opt(activeMode, opt, &len));
    opt += len;
    if (useBasicReg) {
        CHECK_STATUS(set_chars_define_opt("BASIC_REG", opt, &len));
        opt += len;
    }
    if (outputNcwhMode) {
        CHECK_STATUS(set_chars_define_opt("USE_NCWH", opt, &len));
        opt += len;
    }
    return SUCCESS;
}

inline EE set_conv_depthwise_dila_opt_mali(U32 fw,
    U32 fh,
    U32 sw,
    U32 dw,
    U32 workEntriesPerThread,
    ActivationMode activeMode,
    bool outputNcwhMode,
    DataType dt,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    U32 len = 0;
    U32 ON = workEntriesPerThread;
    char modeName[128] = "";
    CHECK_STATUS(set_activation_modeName(activeMode, modeName, &len));
    std::string dilaMode = "dilax_";
    if (dw == 2) {
        dilaMode = "dila2_";
    }
    std::string formatName = "";
    if (outputNcwhMode) {
        formatName = "ncwh_";
    }
    sprintf(kernelName, "conv_depthwise_sw%d_%s%s%s%d%d%d", sw, dilaMode.c_str(), modeName,
        formatName.c_str(), fw, fh, ON);

    sprintf(kernelOpt->sourceName, "conv_depthwise_sw%d_dila", sw);
    kernelOpt->kernelDataType = dt;
    char *opt = kernelOpt->option;
    if (ON < 1 || ON > 8) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    if (dw == 2 && ON <= 2) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    U32 IN = ON;
    U32 FWH = fw * fh;
    U32 LN = IN;
    U32 UN = 0;
    if (dw == 2) {
        if (sw == 2) {
            LN = IN - 1;
            UN = LN;
        } else {
            LN = IN - 2;
        }
    }
    CHECK_STATUS(set_value_define_opt(fw, "FW", opt, &len));
    opt += len;
    CHECK_STATUS(set_value_define_opt(fh, "FH", opt, &len));
    opt += len;
    CHECK_STATUS(set_value_define_opt(ON, "ON", opt, &len));
    opt += len;
    CHECK_STATUS(set_value_define_opt(IN, "IN", opt, &len));
    opt += len;
    CHECK_STATUS(set_value_define_opt(LN, "LN", opt, &len));
    opt += len;
    CHECK_STATUS(set_value_define_opt(FWH, "FWH", opt, &len));
    opt += len;
    CHECK_STATUS(set_activation_define_opt(activeMode, opt, &len));
    opt += len;
    if (dw == 2) {
        if (sw == 2) {
            CHECK_STATUS(set_value_define_opt(UN, "UN", opt, &len));
            opt += len;
        }
        CHECK_STATUS(set_chars_define_opt("DILATION2", opt, &len));
        opt += len;
    }
    if (outputNcwhMode) {
        CHECK_STATUS(set_chars_define_opt("USE_NCWH", opt, &len));
        opt += len;
    }
    return SUCCESS;
}
#endif
