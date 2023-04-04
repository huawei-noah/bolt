#ifndef CONV_DIRECT_OPT
#define CONV_DIRECT_OPT
#include "common_opt.h"

inline EE set_conv_direct_trans_flt(U32 workChannelsPerThread,
    U32 workFiltersPerThread,
    bool transWH,
    DataType dt,
    GCLMemType outputMemType,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    char ioMemName[128] = "";
    U32 item_c = workChannelsPerThread;
    U32 item_k = workFiltersPerThread;
    kernelOpt->kernelDataType = dt;
    char *opt = kernelOpt->option;
    CHECK_STATUS(set_io_mem_name(GCL_MEM_BUF, outputMemType, ioMemName));
    std::string transWHName = "";
    if (transWH) {
        transWHName = "hw_";
        CHECK_STATUS(set_chars_define_opt("USE_TRANS_WH", opt));
    }
    std::string kernel = std::string("conv_direct_trans_flt_") + ioMemName + transWHName +
        std::to_string(item_c) + std::to_string(item_k);
    UNI_STRCPY(kernelName, kernel.c_str());
    UNI_STRCPY(kernelOpt->sourceName, "conv_direct_trans_flt");
    CHECK_STATUS(set_value_define_opt(item_c, "C", opt));
    CHECK_STATUS(set_value_define_opt(item_k, "K", opt));
    CHECK_STATUS(set_io_mem_define_opt(GCL_MEM_BUF, outputMemType, opt));
    return SUCCESS;
}

inline EE get_conv_direct_cal_scheme(
    std::vector<U32> *vh, std::vector<U32> *vc, std::vector<U32> *vk, U32 fh, U32 sh, U32 fn)
{
    U32 in = 2;
    U32 jn = 8;
    U32 k = 4;
    if (fh <= 3 && sh == 1) {
        in = 3;
    }
    for (U32 i = 0; i < in; i++) {
        for (U32 j = 1; j <= jn; j++) {
            vh->push_back(j);
            vc->push_back(4);
            vk->push_back(k);
        }
        k = k << 1;
        if (fn % k != 0) {
            break;
        }
        jn = (k == 8) ? 4 : 3;
    }
    return SUCCESS;
}

inline EE set_conv_direct_opt_mali(U32 fw,
    U32 fh,
    U32 ft,
    U32 sh,
    U32 workEntriesPerThread,
    U32 workFiltersPerThread,
    bool useNoBiasMode,
    ActivationParamSpec activeMode,
    DataType dt,
    GCLMemType inputMemType,
    GCLMemType outputMemType,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    U32 ON = workEntriesPerThread;
    U32 KN = workFiltersPerThread;
    char devName[128] = "";
    bool useQualcomm = check_qualcomm_device(devName);
    char modeName[128] = "";
    CHECK_STATUS(set_activation_mode_name(activeMode, modeName));
    char ioMemName[128] = "";
    CHECK_STATUS(set_io_mem_name(inputMemType, outputMemType, ioMemName));
    std::string biasName = "";
    if (useNoBiasMode) {
        biasName = "nobias_";
    }

    std::string kernel, source;
    if (ft > 1) {
        source = std::string("conv_direct_3d_sh") + std::to_string(sh) + std::string(devName);
        kernel = source + std::string("_") + ioMemName + modeName + biasName + std::to_string(fw) +
            std::to_string(fh) + std::to_string(ft) + std::to_string(ON) + std::to_string(KN);
    } else {
        source = std::string("conv_direct_sh") + std::to_string(sh) + std::string(devName);
        kernel = source + std::string("_") + ioMemName + modeName + biasName + std::to_string(fw) +
            std::to_string(fh) + std::to_string(ON) + std::to_string(KN);
    }
    UNI_STRCPY(kernelName, kernel.c_str());
    UNI_STRCPY(kernelOpt->sourceName, source.c_str());

    kernelOpt->kernelDataType = dt;
    char *opt = kernelOpt->option;
    if (ON < 1 || ON > 8) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    bool useBasicReg = true;
    U32 FWH = fw * fh;
    U32 FWHT = fw * fh * ft;
    U32 in = ON;
    U32 LN = in - 1;
    U32 UN = LN;
    if (!useQualcomm) {
        if (fh == 1) {
            in = ON;
            LN = in;
            UN = 0;
        } else {
            if (sh == 1) {
                if ((fh == 2 && ON < 5) || (fh == 3 && ON < 4) || (fh <= 7 && ON < 3)) {
                    useBasicReg = false;
                }
            } else {
                if (ON <= 2) {
                    useBasicReg = false;
                }
            }
            if (!useBasicReg) {
                in = (ON - 1) * sh + fh;
                LN = in;
                UN = LN - 1;
            }
        }
    }

    CHECK_STATUS(set_value_define_opt(fw, "FW", opt));
    CHECK_STATUS(set_value_define_opt(fh, "FH", opt));
    CHECK_STATUS(set_value_define_opt(ON, "ON", opt));
    CHECK_STATUS(set_value_define_opt(in, "IN", opt));
    CHECK_STATUS(set_value_define_opt(LN, "LN", opt));
    CHECK_STATUS(set_value_define_opt(LN, "UN", opt));
    CHECK_STATUS(set_value_define_opt(KN, "KN", opt));
    if (ft > 1) {
        CHECK_STATUS(set_value_define_opt(FWHT, "FWHT", opt));
        CHECK_STATUS(set_value_define_opt(ft, "FT", opt));
    } else {
        CHECK_STATUS(set_value_define_opt(FWH, "FWH", opt));
    }
    CHECK_STATUS(set_activation_define_opt(activeMode, opt));
    if (useBasicReg) {
        CHECK_STATUS(set_chars_define_opt("BASIC_REG", opt));
    }
    if (useNoBiasMode) {
        CHECK_STATUS(set_chars_define_opt("NO_BIAS", opt));
    }
    CHECK_STATUS(set_io_mem_define_opt(inputMemType, outputMemType, opt));
    return SUCCESS;
}

inline EE get_conv_direct_multi_batch_cal_scheme(
    std::vector<U32> *vh, std::vector<U32> *vc, std::vector<U32> *vk, U32 fn)
{
    for (U32 n = 2; n <= 4; n++) {
        for (U32 k = 4; k <= 8; k += 4) {
            for (U32 w = 1; w <= 3; w++) {
                if (k == 8 && (w > 1 || fn % k != 0)) {
                    continue;
                }
                if (n > 2 && (w * k > 8 || w > 1)) {
                    continue;
                }
                vh->push_back(w + (n << 4));
                vc->push_back(4);
                vk->push_back(k);
            }
        }
    }
    return SUCCESS;
}

inline EE set_conv_direct_multi_batch_opt_mali(U32 fw,
    U32 fh,
    U32 ft,
    U32 sh,
    U32 workEntriesPerThread,
    U32 workFiltersPerThread,
    U32 workBatchesPerThread,
    ActivationParamSpec activeMode,
    DataType dt,
    GCLMemType inputMemType,
    GCLMemType outputMemType,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    U32 ON = workEntriesPerThread;
    U32 KN = workFiltersPerThread;
    U32 BN = workBatchesPerThread;
    char modeName[128] = "";
    CHECK_STATUS(set_activation_mode_name(activeMode, modeName));
    char ioMemName[128] = "";
    CHECK_STATUS(set_io_mem_name(inputMemType, outputMemType, ioMemName));
    std::string source = std::string("conv_direct_multi_batch_sh") + std::to_string(sh);
    std::string kernel = source + std::string("_") + ioMemName + modeName + std::to_string(fw) +
        std::to_string(fh) + std::to_string(ON) + std::to_string(KN) + std::to_string(BN);
    UNI_STRCPY(kernelName, kernel.c_str());
    UNI_STRCPY(kernelOpt->sourceName, source.c_str());
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
    U32 in, LN, UN;
    if (fh == 1) {
        in = ON;
        LN = in;
        UN = 0;
    } else {
        in = ON;
        LN = in - 1;
        UN = LN;
    }

    CHECK_STATUS(set_value_define_opt(fw, "FW", opt));
    CHECK_STATUS(set_value_define_opt(fh, "FH", opt));
    CHECK_STATUS(set_value_define_opt(ON, "ON", opt));
    CHECK_STATUS(set_value_define_opt(in, "IN", opt));
    CHECK_STATUS(set_value_define_opt(LN, "LN", opt));
    CHECK_STATUS(set_value_define_opt(LN, "UN", opt));
    CHECK_STATUS(set_value_define_opt(KN, "KN", opt));
    CHECK_STATUS(set_value_define_opt(BN, "BN", opt));
    CHECK_STATUS(set_value_define_opt(FWH, "FWH", opt));
    CHECK_STATUS(set_activation_define_opt(activeMode, opt));
    if (useBasicReg) {
        CHECK_STATUS(set_chars_define_opt("BASIC_REG", opt));
    }
    CHECK_STATUS(set_io_mem_define_opt(inputMemType, outputMemType, opt));
    return SUCCESS;
}

inline EE get_conv_direct_reuse_w_cal_scheme(std::vector<U32> *vh,
    std::vector<U32> *vc,
    std::vector<U32> *vk,
    U32 iw,
    U32 fn,
    GCLMemType inputMemType)
{
    U32 in = 3;
    //    U32 jn = (inputMemType == GCL_MEM_BUF) ? 2 : 3;
    U32 jn = 2;
    U32 k = 4;
    for (U32 i = 0; i < in; i++) {
        U32 w = 2;
        for (U32 j = 0; j < jn; j++) {
            if (iw % w == 0) {
                vh->push_back(w << 8);
                vc->push_back(4);
                vk->push_back(k);
                //                h = (inputMemType == GCL_MEM_BUF) ? (h << 1) : h++;
                w = w << 1;
            }
        }
        k = k << 1;
        if (fn % k != 0) {
            break;
        }
    }
    return SUCCESS;
}

inline EE set_conv_direct_reuse_w_opt_mali(U32 fw,
    U32 fh,
    U32 ft,
    U32 sw,
    U32 workEntriesPerThread,
    U32 workFiltersPerThread,
    bool useNoBiasMode,
    ActivationParamSpec activeMode,
    DataType dt,
    GCLMemType inputMemType,
    GCLMemType outputMemType,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    U32 ON = workEntriesPerThread;
    U32 KN = workFiltersPerThread;
    char modeName[128] = "";
    CHECK_STATUS(set_activation_mode_name(activeMode, modeName));
    char ioMemName[128] = "";
    CHECK_STATUS(set_io_mem_name(inputMemType, outputMemType, ioMemName));
    std::string kernel = std::string("conv_direct_sw1_reuse_w_") + ioMemName + modeName +
        std::to_string(fw) + std::to_string(fh) + std::to_string(ON) + std::to_string(KN);
    UNI_STRCPY(kernelName, kernel.c_str());
    UNI_STRCPY(kernelOpt->sourceName, "conv_direct_sw1_reuse_w");
    kernelOpt->kernelDataType = dt;
    char *opt = kernelOpt->option;
    if (ON < 1 || ON > 8) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    if (fw != 1 || fh != 1 || ft != 1 || sw != 1) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    U32 in = ON;
    U32 LN = in;

    CHECK_STATUS(set_value_define_opt(fw, "FW", opt));
    CHECK_STATUS(set_value_define_opt(fh, "FH", opt));
    CHECK_STATUS(set_value_define_opt(ON, "ON", opt));
    CHECK_STATUS(set_value_define_opt(in, "IN", opt));
    CHECK_STATUS(set_value_define_opt(LN, "LN", opt));
    CHECK_STATUS(set_value_define_opt(KN, "KN", opt));
    CHECK_STATUS(set_activation_define_opt(activeMode, opt));
    if (useNoBiasMode) {
        CHECK_STATUS(set_chars_define_opt("NO_BIAS", opt));
    }
    CHECK_STATUS(set_io_mem_define_opt(inputMemType, outputMemType, opt));
    return SUCCESS;
}

inline EE get_conv_direct_nchw_to_nchwc4_cal_scheme(
    std::vector<U32> *vh, std::vector<U32> *vc, std::vector<U32> *vk, U32 fw, U32 sw)
{
    U32 in = 8;
    U32 i_be = 2;
    U32 i_str = 1;
    if (check_qualcomm_device()) {
        in = (16 * sw - fw) / sw + 1;
        if (in > 12) {
            in = 12;
        }
        i_be = 4 / sw;
        i_str = 4 / sw;
    }
    for (U32 i = i_be; i <= in; i += i_str) {
        vh->push_back(i);
        vc->push_back(1);
        vk->push_back(4);
    }
    return SUCCESS;
}

inline EE set_conv_direct_nchw_to_nchwc4_opt_mali(U32 fw,
    U32 fh,
    U32 ft,
    U32 sw,
    U32 workEntriesPerThread,
    ActivationParamSpec activeMode,
    DataType dt,
    GCLMemType inputMemType,
    GCLMemType outputMemType,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    U32 ON = workEntriesPerThread;
    char devName[128] = "";
    bool useQualcomm = check_qualcomm_device(devName);
    char modeName[128] = "";
    CHECK_STATUS(set_activation_mode_name(activeMode, modeName));
    char ioMemName[128] = "";
    CHECK_STATUS(set_io_mem_name(inputMemType, outputMemType, ioMemName));
    std::string kernel, source;
    if (ft > 1) {
        source = std::string("conv_direct_3d_sw") + std::to_string(sw) +
            std::string("_nchw_to_nchwc4") + devName;
        kernel = source + std::string("_") + ioMemName + modeName + std::to_string(fw) +
            std::to_string(fh) + std::to_string(ft) + std::to_string(ON);
    } else {
        source = std::string("conv_direct_sw") + std::to_string(sw) +
            std::string("_nchw_to_nchwc4") + devName;
        kernel = source + std::string("_") + ioMemName + modeName + std::to_string(fw) +
            std::to_string(fh) + std::to_string(ON);
    }
    UNI_STRCPY(kernelName, kernel.c_str());
    UNI_STRCPY(kernelOpt->sourceName, source.c_str());

    kernelOpt->kernelDataType = dt;
    char *opt = kernelOpt->option;
    if (fw > 7 || fh > 7) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    U32 in, FWH, FWHT;
    in = (ON - 1) * sw + fw;
    FWH = fw * fh;
    FWHT = fw * fh * ft;
    if (in < 1 || in > 16 * sw) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    CHECK_STATUS(set_value_define_opt(fw, "FW", opt));
    CHECK_STATUS(set_value_define_opt(fh, "FH", opt));
    CHECK_STATUS(set_value_define_opt(ON, "ON", opt));
    CHECK_STATUS(set_value_define_opt(in, "IN", opt));
    if (ft == 1) {
        CHECK_STATUS(set_value_define_opt(FWH, "FWH", opt));
    } else {
        CHECK_STATUS(set_value_define_opt(ft, "FT", opt));
        CHECK_STATUS(set_value_define_opt(FWHT, "FWHT", opt));
    }
    CHECK_STATUS(set_activation_define_opt(activeMode, opt));
    CHECK_STATUS(set_io_mem_define_opt(inputMemType, outputMemType, opt));
    return SUCCESS;
}

inline EE get_conv_direct_dila_cal_scheme(
    std::vector<U32> *vh, std::vector<U32> *vc, std::vector<U32> *vk, U32 dh, U32 fn)
{
    U32 in = 2;
    U32 jn = 8;
    U32 j_be = (dh == 2) ? 3 : 1;
    U32 k = 4;
    for (U32 i = 0; i < in; i++) {
        for (U32 j = j_be; j <= jn; j++) {
            vh->push_back(j);
            vc->push_back(4);
            vk->push_back(k);
        }
        k = k << 1;
        if (fn % k != 0) {
            break;
        }
        jn = (k == 8) ? 4 : 3;
    }
    return SUCCESS;
}

inline EE set_conv_direct_dila_opt_mali(U32 fw,
    U32 fh,
    U32 sh,
    U32 dh,
    U32 workEntriesPerThread,
    U32 workFiltersPerThread,
    ActivationParamSpec activeMode,
    DataType dt,
    GCLMemType inputMemType,
    GCLMemType outputMemType,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    U32 ON = workEntriesPerThread;
    U32 KN = workFiltersPerThread;
    char devName[128] = "";
    if (dh == 2 || sh == 2) {
        check_qualcomm_device(devName);
    }
    char modeName[128] = "";
    CHECK_STATUS(set_activation_mode_name(activeMode, modeName));
    char ioMemName[128] = "";
    CHECK_STATUS(set_io_mem_name(inputMemType, outputMemType, ioMemName));
    std::string dilaMode = "dilax_";
    if (dh == 2) {
        dilaMode = "dila2_";
    }
    std::string kernel = std::string("conv_direct_sh") + std::to_string(sh) + devName +
        std::string("_") + dilaMode + ioMemName + modeName + std::to_string(fw) +
        std::to_string(fh) + std::to_string(ON) + std::to_string(KN);
    std::string source =
        std::string("conv_direct_sh") + std::to_string(sh) + devName + std::string("_dila");
    UNI_STRCPY(kernelName, kernel.c_str());
    UNI_STRCPY(kernelOpt->sourceName, source.c_str());
    kernelOpt->kernelDataType = dt;
    char *opt = kernelOpt->option;
    if (ON < 1 || ON > 8) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    if (dh == 2 && ON <= 2) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    U32 in = ON;
    U32 FWH = fw * fh;
    U32 LN = in;
    U32 UN = 0;
    if (dh == 2) {
        if (sh == 2) {
            LN = in - 1;
            UN = LN;
        } else {
            LN = in - 2;
        }
    }

    CHECK_STATUS(set_value_define_opt(fw, "FW", opt));
    CHECK_STATUS(set_value_define_opt(fh, "FH", opt));
    CHECK_STATUS(set_value_define_opt(ON, "ON", opt));
    CHECK_STATUS(set_value_define_opt(in, "IN", opt));
    CHECK_STATUS(set_value_define_opt(LN, "LN", opt));
    CHECK_STATUS(set_value_define_opt(KN, "KN", opt));
    CHECK_STATUS(set_value_define_opt(FWH, "FWH", opt));
    CHECK_STATUS(set_activation_define_opt(activeMode, opt));
    if (dh == 2) {
        if (sh == 2) {
            CHECK_STATUS(set_value_define_opt(UN, "UN", opt));
        }
        CHECK_STATUS(set_chars_define_opt("DILATION2", opt));
    }
    CHECK_STATUS(set_io_mem_define_opt(inputMemType, outputMemType, opt));
    return SUCCESS;
}

inline EE get_conv_direct_sh1_fn_spe_cal_scheme(
    std::vector<U32> *vh, std::vector<U32> *vc, std::vector<U32> *vk, U32 fh, GCLMemType outputMemType)
{
    U32 item_h = 8;
    if (fh >= 7) {
        item_h = (outputMemType == GCL_MEM_BUF) ? 6 : 4;
    }
    vh->push_back(item_h);
    vc->push_back(4);
    vk->push_back(1);
    return SUCCESS;
}

inline EE set_conv_direct_sh1_fn_spe_opt_mali(U32 fw,
    U32 fh,
    U32 workEntriesPerThread,
    bool useNchwFormat,
    ActivationParamSpec activeMode,
    DataType dt,
    GCLMemType inputMemType,
    GCLMemType outputMemType,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    U32 ON = workEntriesPerThread;
    char modeName[128] = "";
    CHECK_STATUS(set_activation_mode_name(activeMode, modeName));
    char ioMemName[128] = "";
    CHECK_STATUS(set_io_mem_name(inputMemType, outputMemType, ioMemName));
    std::string formatName = "";
    if (useNchwFormat) {
        formatName = "nchw_";
    }
    std::string buffer = std::string("conv_direct_sh1_fn_spe_") + ioMemName + modeName +
        formatName + std::to_string(fw) + std::to_string(fh) + std::to_string(ON);
    UNI_STRCPY(kernelName, buffer.c_str());
    UNI_STRCPY(kernelOpt->sourceName, "conv_direct_sh1_fn_spe");
    kernelOpt->kernelDataType = dt;
    char *opt = kernelOpt->option;

    U32 in = ON - 1 + fh;
    U32 LN = in;
    U32 UN = LN - 1;
    CHECK_STATUS(set_value_define_opt(fw, "FW", opt));
    CHECK_STATUS(set_value_define_opt(fh, "FH", opt));
    CHECK_STATUS(set_value_define_opt(ON, "ON", opt));
    CHECK_STATUS(set_value_define_opt(in, "IN", opt));
    CHECK_STATUS(set_value_define_opt(LN, "LN", opt));
    CHECK_STATUS(set_value_define_opt(UN, "UN", opt));
    CHECK_STATUS(set_activation_define_opt(activeMode, opt));
    if (useNchwFormat) {
        CHECK_STATUS(set_chars_define_opt("USE_NCHW", opt));
    }
    CHECK_STATUS(set_io_mem_define_opt(inputMemType, outputMemType, opt));
    return SUCCESS;
}

#endif
