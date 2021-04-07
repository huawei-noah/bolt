#ifndef CONV_DIRECT_OPT
#define CONV_DIRECT_OPT
#include "common_opt.h"
inline EE set_conv_direct_opt_mali(U32 fw,
    U32 fh,
    U32 ft,
    U32 sw,
    U32 workEntriesPerThread,
    U32 workFiltersPerThread,
    ActivationMode activeMode,
    DataType dt,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    U32 len = 0;
    U32 ON = workEntriesPerThread;
    U32 KN = workFiltersPerThread;
    char modeName[128] = "";
    CHECK_STATUS(set_activation_modeName(activeMode, modeName, &len));

    sprintf(kernelName, "conv_direct_sw%d_%s%d%d%d%d", sw, modeName, fw, fh, ON, KN);
    sprintf(kernelOpt->sourceName, "conv_direct_sw%d", sw);
    kernelOpt->kernelDataType = dt;
    char *opt = kernelOpt->option;
    if (ON < 1 || ON > 8) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    if (ft > 1) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    bool useBasicReg = true;
    U32 FWH = fw * fh;
    U32 IN, LN, UN;
    if (fw == 1) {
        IN = ON;
        LN = IN;
        UN = 0;
    } else {
        if (sw == 1) {
            if ((fw == 2 && ON < 5) || (fw == 3 && ON < 4) || (fw <= 7 && ON < 3)) {
                useBasicReg = false;
            }
        } else {
            if (ON <= 2) {
                useBasicReg = false;
            }
        }
        IN = ON;
        LN = IN - 1;
        UN = LN;
        if (!useBasicReg) {
            IN = (ON - 1) * sw + fw;
            LN = IN;
            UN = LN - 1;
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
    CHECK_STATUS(set_value_define_opt(LN, "UN", opt, &len));
    opt += len;
    CHECK_STATUS(set_value_define_opt(KN, "KN", opt, &len));
    opt += len;
    CHECK_STATUS(set_value_define_opt(FWH, "FWH", opt, &len));
    opt += len;
    CHECK_STATUS(set_activation_define_opt(activeMode, opt, &len));
    opt += len;
    if (useBasicReg) {
        CHECK_STATUS(set_chars_define_opt("BASIC_REG", opt, &len));
        opt += len;
    }
    return SUCCESS;
}

inline EE set_conv_direct_multi_batch_opt_mali(U32 fw,
    U32 fh,
    U32 ft,
    U32 sw,
    U32 workEntriesPerThread,
    U32 workFiltersPerThread,
    U32 workBatchesPerThread,
    ActivationMode activeMode,
    DataType dt,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    U32 len = 0;
    U32 ON = workEntriesPerThread;
    U32 KN = workFiltersPerThread;
    U32 BN = workBatchesPerThread;
    char modeName[128] = "";
    CHECK_STATUS(set_activation_modeName(activeMode, modeName, &len));

    sprintf(
        kernelName, "conv_direct_multi_batch_sw%d_%s%d%d%d%d%d", sw, modeName, fw, fh, ON, KN, BN);
    sprintf(kernelOpt->sourceName, "conv_direct_multi_batch_sw%d", sw);
    kernelOpt->kernelDataType = dt;
    char *opt = kernelOpt->option;
    if (ON < 1 || ON > 8) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    if (ft > 1) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    bool useBasicReg = true;
    U32 FWH = fw * fh;
    U32 IN, LN, UN;
    if (fw == 1) {
        IN = ON;
        LN = IN;
        UN = 0;
    } else {
        IN = ON;
        LN = IN - 1;
        UN = LN;
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
    CHECK_STATUS(set_value_define_opt(KN, "KN", opt, &len));
    opt += len;
    CHECK_STATUS(set_value_define_opt(BN, "BN", opt, &len));
    opt += len;
    CHECK_STATUS(set_value_define_opt(FWH, "FWH", opt, &len));
    opt += len;
    CHECK_STATUS(set_activation_define_opt(activeMode, opt, &len));
    opt += len;
    if (useBasicReg) {
        CHECK_STATUS(set_chars_define_opt("BASIC_REG", opt, &len));
        opt += len;
    }
    return SUCCESS;
}

inline EE set_conv_direct_spe_fwhs1_opt_mali(U32 fw,
    U32 fh,
    U32 ft,
    U32 sw,
    U32 workChannelEntriesPerLoop,
    bool useNoBiasMode,
    bool useGemvMode,
    ActivationMode activeMode,
    DataType dt,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    U32 len = 0;
    U32 OC = workChannelEntriesPerLoop;
    char modeName[128] = "";
    CHECK_STATUS(set_activation_modeName(activeMode, modeName, &len));
    std::string gemvName = "";
    std::string noBiasName = "";
    if (useGemvMode) {
        gemvName = "gemv_";
    }
    if (useNoBiasMode) {
        noBiasName = "nobias_";
    }

    sprintf(kernelName, "conv_direct_spe_fwhs1_%s%s%s%d", modeName, gemvName.c_str(),
        noBiasName.c_str(), OC);
    sprintf(kernelOpt->sourceName, "conv_direct_spe_fwhs1");
    kernelOpt->kernelDataType = dt;
    char *opt = kernelOpt->option;
    if (fw != 1 || fh != 1 || ft != 1 || sw != 1) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    CHECK_STATUS(set_value_define_opt(OC, "OC", opt, &len));
    opt += len;
    CHECK_STATUS(set_activation_define_opt(activeMode, opt, &len));
    opt += len;
    if (useNoBiasMode) {
        CHECK_STATUS(set_chars_define_opt("NO_BIAS", opt, &len));
        opt += len;
    }
    if (useGemvMode) {
        CHECK_STATUS(set_chars_define_opt("USE_GEMV", opt, &len));
        opt += len;
    }

    return SUCCESS;
}

inline EE set_conv_direct_reuse_h_opt_mali(U32 fw,
    U32 fh,
    U32 ft,
    U32 sh,
    U32 workEntriesPerThread,
    U32 workFiltersPerThread,
    ActivationMode activeMode,
    DataType dt,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    U32 len = 0;
    U32 ON = workEntriesPerThread;
    U32 KN = workFiltersPerThread;
    char modeName[128] = "";
    CHECK_STATUS(set_activation_modeName(activeMode, modeName, &len));

    sprintf(kernelName, "conv_direct_sh1_reuse_h_%s%d%d%d%d", modeName, fw, fh, ON, KN);
    sprintf(kernelOpt->sourceName, "conv_direct_sh1_reuse_h");
    kernelOpt->kernelDataType = dt;
    char *opt = kernelOpt->option;
    if (ON < 1 || ON > 8) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    if (fw != 1 || fh != 1 || ft != 1 || sh != 1) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    U32 IN = ON;
    U32 LN = IN;

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
    CHECK_STATUS(set_value_define_opt(KN, "KN", opt, &len));
    opt += len;
    CHECK_STATUS(set_activation_define_opt(activeMode, opt, &len));
    opt += len;
    return SUCCESS;
}

inline EE set_conv_direct_ncwh_to_nchwc4_opt_mali(U32 fw,
    U32 fh,
    U32 ft,
    U32 sw,
    U32 workEntriesPerThread,
    ActivationMode activeMode,
    DataType dt,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    U32 len = 0;
    U32 ON = workEntriesPerThread;
    char modeName[128] = "";
    CHECK_STATUS(set_activation_modeName(activeMode, modeName, &len));
    if (ft > 1) {
        sprintf(kernelName, "conv_direct_3d_sw%d_nchw_to_ncwhc4_%s%d%d%d%d", sw, modeName, fw, fh,
            ft, ON);
        sprintf(kernelOpt->sourceName, "conv_direct_3d_sw%d_nchw_to_ncwhc4", sw);
    } else {
        sprintf(kernelName, "conv_direct_sw%d_nchw_to_ncwhc4_%s%d%d%d", sw, modeName, fw, fh, ON);
        sprintf(kernelOpt->sourceName, "conv_direct_sw%d_nchw_to_ncwhc4", sw);
    }

    kernelOpt->kernelDataType = dt;
    char *opt = kernelOpt->option;
    if (ON < 1 || ON > 8) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    if (fw > 7 || fh > 7) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    U32 IN, FWH, FWHT;
    IN = (ON - 1) * sw + fw;
    FWH = fw * fh;
    FWHT = fw * fh * ft;

    CHECK_STATUS(set_value_define_opt(fw, "FW", opt, &len));
    opt += len;
    CHECK_STATUS(set_value_define_opt(fh, "FH", opt, &len));
    opt += len;
    CHECK_STATUS(set_value_define_opt(ON, "ON", opt, &len));
    opt += len;
    CHECK_STATUS(set_value_define_opt(IN, "IN", opt, &len));
    opt += len;
    if (ft == 1) {
        CHECK_STATUS(set_value_define_opt(FWH, "FWH", opt, &len));
        opt += len;
    } else {
        CHECK_STATUS(set_value_define_opt(ft, "FT", opt, &len));
        opt += len;
        CHECK_STATUS(set_value_define_opt(FWHT, "FWHT", opt, &len));
        opt += len;
    }
    CHECK_STATUS(set_activation_define_opt(activeMode, opt, &len));
    opt += len;
    return SUCCESS;
}

inline EE set_conv_direct_dila_opt_mali(U32 fw,
    U32 fh,
    U32 sw,
    U32 dw,
    U32 workEntriesPerThread,
    U32 workFiltersPerThread,
    ActivationMode activeMode,
    DataType dt,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    U32 len = 0;
    U32 ON = workEntriesPerThread;
    U32 KN = workFiltersPerThread;
    char modeName[128] = "";
    CHECK_STATUS(set_activation_modeName(activeMode, modeName, &len));
    std::string dilaMode = "dilax_";
    if (dw == 2) {
        dilaMode = "dila2_";
    }
    sprintf(kernelName, "conv_direct_sw%d_%s%s%d%d%d%d", sw, dilaMode.c_str(), modeName, fw, fh, ON,
        KN);
    sprintf(kernelOpt->sourceName, "conv_direct_sw%d_dila", sw);
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
    CHECK_STATUS(set_value_define_opt(KN, "KN", opt, &len));
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
    return SUCCESS;
}

#endif
