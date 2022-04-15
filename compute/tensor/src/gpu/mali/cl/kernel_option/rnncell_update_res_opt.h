#ifndef RNNCELL_UPDATE_RES_OPT
#define RNNCELL_UPDATE_RES_OPT
#include "common_opt.h"
inline EE set_rnncell_update_res_opt_mali(bool useProjection,
    bool useRnnMode,
    DataType dt,
    GCLMemType inputMemType,
    GCLMemType outputMemType,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    char *opt = kernelOpt->option;
    char ioMemName[128] = "";
    CHECK_STATUS(set_io_mem_name(inputMemType, outputMemType, ioMemName));
    std::string proName = "";
    if (useProjection) {
        proName = "pro_";
        CHECK_STATUS(set_chars_define_opt("USE_PROJECTION", opt));
    }
    std::string modeName = "s";
    if (useRnnMode) {
        modeName = "ch";
        CHECK_STATUS(set_chars_define_opt("USE_RNN_MODE", opt));
    }
    kernelOpt->kernelDataType = dt;
    std::string kernel = std::string("rnncell_update_res_") + proName + modeName;
    UNI_STRCPY(kernelName, kernel.c_str());
    UNI_STRCPY(kernelOpt->sourceName, "rnncell_update_res");
    CHECK_STATUS(set_io_mem_define_opt(inputMemType, outputMemType, opt));
    return SUCCESS;
}
#endif
