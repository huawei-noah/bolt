#ifndef DECONV_OPT
#define DECONV_OPT
#include "common_opt.h"
inline EE set_deconv_gemm_trans_fltbuf(U32 workChannelsPerThread,
    U32 workFiltersPerThread,
    DataType dt,
    GCLMemType outputMemType,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    char ioMemName[128] = "";
    U32 item_c = workChannelsPerThread >> 2;
    U32 item_k = workFiltersPerThread;
    kernelOpt->kernelDataType = dt;
    char *opt = kernelOpt->option;
    CHECK_STATUS(set_io_mem_name(GCL_MEM_BUF, outputMemType, ioMemName));

    std::string kernel =
        std::string("deconv_gemm_trans_fltbuf_") + std::to_string(item_c) + std::to_string(item_k);
    UNI_STRCPY(kernelName, kernel.c_str());
    UNI_STRCPY(kernelOpt->sourceName, "deconv_gemm_trans_fltbuf");
    CHECK_STATUS(set_value_define_opt(item_c, "C", opt));
    CHECK_STATUS(set_value_define_opt(item_k, "K", opt));
    CHECK_STATUS(set_io_mem_define_opt(GCL_MEM_BUF, outputMemType, opt));
    return SUCCESS;
}

inline EE get_deconv_gemm_f2s2_scheme(
    std::vector<U32> *vh, std::vector<U32> *vc, std::vector<U32> *vk, U32 iw)
{
    U32 c = 8;
    U32 in = 4;
    for (U32 ii = 0; ii < 2; ii++) {
        for (U32 i = 0; i < in; i++) {
            vh->push_back(i + 1);
            vc->push_back(c);
            vk->push_back(4);
        }
        c = c << 1;
        in = 3;
    }

    if (!check_qualcomm_device()) {
        in = 2;
        U32 w = 2;
        for (U32 ii = 0; ii < 2; ii++) {
            c = 8;
            if (iw % w == 0) {
                for (U32 i = 0; i < in; i++) {
                    vh->push_back(w << 8);
                    vc->push_back(c);
                    vk->push_back(4);
                    c = c << 1;
                }
            }
            w = w << 1;
            in = 1;
        }
    }
    return SUCCESS;
}

inline EE get_deconv_gemm_scheme(
    std::vector<U32> *vh, std::vector<U32> *vc, std::vector<U32> *vk, U32 iw, U32 fw, U32 fh, U32 fc)
{
    U32 c = 4;
    U32 in = 8;
    for (U32 ii = 0; ii < 3; ii++) {
        for (U32 i = 0; i < in; i++) {
            vh->push_back(i + 1);
            vc->push_back(c);
            vk->push_back(4);
        }
        c = c << 1;
        if ((fc * fh * fw) % c != 0) {
            break;
        }
        if (ii == 0) {
            in = 4;
        }
        if (ii == 1) {
            in = 3;
        }
    }

    if (!check_qualcomm_device()) {
        c = 4;
        in = 2;
        for (U32 ii = 0; ii < 3; ii++) {
            U32 w = 2;
            if (ii == 2) {
                in = 1;
            }
            for (U32 i = 0; i < in; i++) {
                if (iw % w != 0) {
                    continue;
                }
                vh->push_back(w << 8);
                vc->push_back(c);
                vk->push_back(4);
                w = w << 1;
            }
            c = c << 1;
            if ((fc * fh * fw) % c != 0) {
                break;
            }
        }
    }
    return SUCCESS;
}

inline EE set_deconv_gemm_f2s2_opt(U32 workChannelsPerThread,
    U32 workEntriesPerThread,
    bool reuseOnW,
    ActivationParamSpec activeMode,
    DataType dt,
    GCLMemType inputMemType,
    GCLMemType outputMemType,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    U32 ON = workEntriesPerThread;
    U32 KN = workChannelsPerThread;
    kernelOpt->kernelDataType = dt;
    char *opt = kernelOpt->option;
    char devName[128] = "";
    bool useQualcomm = check_qualcomm_device(devName);
    char modeName[128] = "";
    CHECK_STATUS(set_activation_mode_name(activeMode, modeName));
    char ioMemName[128] = "";
    CHECK_STATUS(set_io_mem_name(inputMemType, outputMemType, ioMemName));
    std::string reuseOnWName = "";
    if (reuseOnW) {
        reuseOnWName = "w_";
    }
    std::string source = std::string("deconv_gemm_f2s2") + devName;
    std::string kernel = source + std::string("_") + reuseOnWName + ioMemName + modeName +
        std::to_string(ON) + std::to_string(KN);
    UNI_STRCPY(kernelName, kernel.c_str());
    UNI_STRCPY(kernelOpt->sourceName, source.c_str());
    U32 in = ON;
    U32 LN = ON;
    CHECK_STATUS(set_value_define_opt(ON, "ON", opt));
    CHECK_STATUS(set_value_define_opt(in, "IN", opt));
    CHECK_STATUS(set_value_define_opt(LN, "LN", opt));
    CHECK_STATUS(set_value_define_opt(KN, "KN", opt));
    CHECK_STATUS(set_activation_define_opt(activeMode, opt));
    if (reuseOnW) {
        CHECK_STATUS(set_chars_define_opt("REUSE_W", opt));
    }
    CHECK_STATUS(set_io_mem_define_opt(inputMemType, outputMemType, opt));
    return SUCCESS;
}

#endif
