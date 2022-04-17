#ifndef MEM_TRANS_OPT
#define MEM_TRANS_OPT
#include "common_opt.h"
typedef enum {
    NCHW_TO_NCHWC4 = 0,
    NCHWC4_TO_NCHW = 1,
    NCHW_TO_NCHW = 2,
    NCHWC4_TO_NCHWC4 = 3
} MemTransFormType;

inline EE set_mem_trans_opt_mali(MemTransFormType type,
    bool use3dMode,
    DataType dt,
    GCLMemType inputMemType,
    GCLMemType outputMemType,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    char *opt = kernelOpt->option;
    char ioMemName[128] = "";
    CHECK_STATUS(set_io_mem_name(inputMemType, outputMemType, ioMemName));
    std::string inputFormat = "nchw_";
    std::string outputFormat = "to_nchw";
    std::string use3dFormat = (use3dMode) ? "3d_" : "";
    switch (type) {
        case NCHW_TO_NCHW:
            CHECK_STATUS(set_chars_define_opt("USE_INPUT_NCHW", opt));
            CHECK_STATUS(set_chars_define_opt("USE_OUTPUT_NCHW", opt));
            break;
        case NCHW_TO_NCHWC4:
            CHECK_STATUS(set_chars_define_opt("USE_INPUT_NCHW", opt));
            CHECK_STATUS(set_chars_define_opt("USE_OUTPUT_NCHWC4", opt));
            outputFormat = "to_nchwc4";
            break;
        case NCHWC4_TO_NCHW:
            CHECK_STATUS(set_chars_define_opt("USE_INPUT_NCHWC4", opt));
            CHECK_STATUS(set_chars_define_opt("USE_OUTPUT_NCHW", opt));
            inputFormat = "nchwc4_";
            break;
        case NCHWC4_TO_NCHWC4:
            CHECK_STATUS(set_chars_define_opt("USE_INPUT_NCHWC4", opt));
            CHECK_STATUS(set_chars_define_opt("USE_OUTPUT_NCHWC4", opt));
            inputFormat = "nchwc4_";
            outputFormat = "to_nchwc4";
            break;
        default:
            CHECK_STATUS(NOT_MATCH);
    }
    std::string kernel =
        std::string("mem_trans_") + use3dFormat + ioMemName + inputFormat + outputFormat;
    UNI_STRCPY(kernelName, kernel.c_str());
    if (use3dMode) {
        UNI_STRCPY(kernelOpt->sourceName, "mem_trans_3d");
    } else {
        UNI_STRCPY(kernelOpt->sourceName, "mem_trans");
    }
    kernelOpt->kernelDataType = dt;
    CHECK_STATUS(set_io_mem_define_opt(inputMemType, outputMemType, opt));
    return SUCCESS;
}

typedef enum {
    C1_TO_C4 = 0,
    C4_TO_C1 = 1,
} MemTransCType;

inline EE set_mem_trans_c_opt_mali(MemTransCType type,
    DataType dt,
    GCLMemType inputMemType,
    GCLMemType outputMemType,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    char *opt = kernelOpt->option;
    char ioMemName[128] = "";
    CHECK_STATUS(set_io_mem_name(inputMemType, outputMemType, ioMemName));
    std::string transFormat = "";
    if (type == C1_TO_C4) {
        transFormat = "c1_to_c4";
        CHECK_STATUS(set_chars_define_opt("TRANS_C1_TO_C4", opt));
    } else if (type == C4_TO_C1) {
        transFormat = "c4_to_c1";
        CHECK_STATUS(set_chars_define_opt("TRANS_C4_TO_C1", opt));
    } else {
        CHECK_STATUS(NOT_MATCH);
    }
    std::string kernel = std::string("mem_trans_c_") + ioMemName + transFormat;
    UNI_STRCPY(kernelName, kernel.c_str());
    UNI_STRCPY(kernelOpt->sourceName, "mem_trans_c");
    kernelOpt->kernelDataType = dt;
    CHECK_STATUS(set_io_mem_define_opt(inputMemType, outputMemType, opt));
    return SUCCESS;
}
#endif
