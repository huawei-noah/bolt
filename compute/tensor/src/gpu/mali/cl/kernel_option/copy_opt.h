#ifndef COPY_OPT
#define COPY_OPT
#include "common_opt.h"
inline EE set_copy_opt_mali(bool useBlockIndex, DataType dt, char *kernelName, KernelOpt *kernelOpt)
{
    std::string BINDName = "";
    std::string dtName = "";
    char *opt = kernelOpt->option;
    if (useBlockIndex) {
        BINDName = "with_block_index_";
    }
    switch (dt) {
        case DT_I32:
            dtName = "i32";
            CHECK_STATUS(set_chars_define_opt(" DT=i32 ", opt));
            break;
        case DT_U32:
            dtName = "u32";
            CHECK_STATUS(set_chars_define_opt(" DT=u32 ", opt));
            break;
            break;
        case DT_F16:
            dtName = "f16";
            CHECK_STATUS(set_chars_define_opt(" DT=f16 ", opt));
            break;
            break;
        default:
            CHECK_STATUS(NOT_SUPPORTED);
    }

    std::string kernel = std::string("copy_") + BINDName + dtName;
    UNI_STRCPY(kernelName, kernel.c_str());
    UNI_STRCPY(kernelOpt->sourceName, "copy");
    kernelOpt->kernelDataType = dt;
    if (useBlockIndex) {
        CHECK_STATUS(set_chars_define_opt("USE_BLOCK_INDEX", opt));
    }
    return SUCCESS;
}

#endif
