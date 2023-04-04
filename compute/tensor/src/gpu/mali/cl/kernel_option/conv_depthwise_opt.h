#ifndef CONV_DEPTHWISE_OPT
#define CONV_DEPTHWISE_OPT
#include "common_opt.h"
inline EE set_conv_depthwise_trans_flt(U32 workFiltersPerThread,
    DataType dt,
    GCLMemType outputMemType,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    char ioMemName[128] = "";
    U32 item_k = workFiltersPerThread;
    kernelOpt->kernelDataType = dt;
    char *opt = kernelOpt->option;
    CHECK_STATUS(set_io_mem_name(GCL_MEM_BUF, outputMemType, ioMemName));
    std::string kernel =
        std::string("conv_depthwise_trans_fltbuf_") + ioMemName + std::to_string(item_k);
    UNI_STRCPY(kernelName, kernel.c_str());
    UNI_STRCPY(kernelOpt->sourceName, "conv_depthwise_trans_fltbuf");
    CHECK_STATUS(set_value_define_opt(item_k, "K", opt));
    CHECK_STATUS(set_io_mem_define_opt(GCL_MEM_BUF, outputMemType, opt));
    return SUCCESS;
}

inline EE get_conv_depthwise_cal_scheme(
    std::vector<U32> *vh, std::vector<U32> *vc, std::vector<U32> *vk)
{
    U32 in = 8;
    for (U32 i = 1; i <= in; i++) {
        vh->push_back(i);
        vc->push_back(1);
        vk->push_back(4);
    }
    return SUCCESS;
}

inline EE set_conv_depthwise_opt_mali(U32 fw,
    U32 fh,
    U32 sh,
    U32 workEntriesPerThread,
    ActivationParamSpec activeMode,
    bool outputNchwMode,
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
    std::string formatName = "";
    if (outputNchwMode) {
        formatName = "nchw_";
    }
    std::string source = std::string("conv_depthwise_sh") + std::to_string(sh) + devName;
    std::string kernel = source + std::string("_") + ioMemName + modeName + formatName +
        std::to_string(fw) + std::to_string(fh) + std::to_string(ON);
    UNI_STRCPY(kernelName, kernel.c_str());
    UNI_STRCPY(kernelOpt->sourceName, source.c_str());
    kernelOpt->kernelDataType = dt;
    char *opt = kernelOpt->option;
    if (ON < 1 || ON > 8) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    U32 in, LN, UN, FWH;
    bool useBasicReg = true;
    in = ON;
    LN = in - 1;
    UN = LN;
    FWH = fw * fh;
    if (!useQualcomm) {
        if (sh == 1) {
            if (fh < 3) {
                useBasicReg = false;
            } else if (fh <= 3 && ON <= 5) {
                useBasicReg = false;
            } else if (fh <= 5 && ON <= 3) {
                useBasicReg = false;
            } else if (fh <= 7 && ON <= 2) {
                useBasicReg = false;
            }
        } else if (sh == 2) {
            if (fh <= 3 && ON <= 3) {
                useBasicReg = false;
            }
        } else {
            CHECK_STATUS(NOT_SUPPORTED);
        }
        if (!useBasicReg) {
            in = (ON - 1) * sh + fh;
            LN = in;
            UN = LN - 1;
        }
    }

    CHECK_STATUS(set_value_define_opt(fw, "FW", opt));
    CHECK_STATUS(set_value_define_opt(fh, "FH", opt));
    CHECK_STATUS(set_value_define_opt(ON, "ON", opt));
    CHECK_STATUS(set_value_define_opt(in, "IN", opt));
    CHECK_STATUS(set_value_define_opt(LN, "LN", opt));
    CHECK_STATUS(set_value_define_opt(UN, "UN", opt));
    CHECK_STATUS(set_value_define_opt(FWH, "FWH", opt));
    CHECK_STATUS(set_activation_define_opt(activeMode, opt));
    if (useBasicReg) {
        CHECK_STATUS(set_chars_define_opt("BASIC_REG", opt));
    }
    if (outputNchwMode) {
        CHECK_STATUS(set_chars_define_opt("USE_NCHW", opt));
    }
    CHECK_STATUS(set_io_mem_define_opt(inputMemType, outputMemType, opt));
    return SUCCESS;
}

inline EE get_conv_depthwise_dila_cal_scheme(
    U32 dh, std::vector<U32> *vh, std::vector<U32> *vc, std::vector<U32> *vk)
{
    U32 in = 8;
    U32 ib = (dh == 2) ? 3 : 1;
    for (U32 i = ib; i <= in; i++) {
        vh->push_back(i);
        vc->push_back(1);
        vk->push_back(4);
    }
    return SUCCESS;
}

inline EE set_conv_depthwise_dila_opt_mali(U32 fw,
    U32 fh,
    U32 sh,
    U32 dh,
    U32 workEntriesPerThread,
    ActivationParamSpec activeMode,
    bool outputNchwMode,
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
    std::string dilaMode = "dilax_";
    if (dh == 2) {
        dilaMode = "dila2_";
    }
    std::string formatName = "";
    if (outputNchwMode) {
        formatName = "nchw_";
    }
    std::string kernel = std::string("conv_depthwise_sh") + std::to_string(sh) + std::string("_") +
        dilaMode + ioMemName + modeName + formatName + std::to_string(fw) + std::to_string(fh) +
        std::to_string(ON);
    UNI_STRCPY(kernelName, kernel.c_str());
    std::string source =
        std::string("conv_depthwise_sh") + std::to_string(sh) + std::string("_dila");
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
    CHECK_STATUS(set_value_define_opt(FWH, "FWH", opt));
    CHECK_STATUS(set_activation_define_opt(activeMode, opt));
    if (dh == 2) {
        if (sh == 2) {
            CHECK_STATUS(set_value_define_opt(UN, "UN", opt));
        }
        CHECK_STATUS(set_chars_define_opt("DILATION2", opt));
    }
    if (outputNchwMode) {
        CHECK_STATUS(set_chars_define_opt("USE_NCHW", opt));
    }
    CHECK_STATUS(set_io_mem_define_opt(inputMemType, outputMemType, opt));
    return SUCCESS;
}
#endif
