#ifndef EXPAND_OPT
#define EXPAND_OPT
#include "common_opt.h"
inline EE set_expand_opt_mali(U32 nDims,
    DataType dt,
    GCLMemType inputMemType,
    GCLMemType outputMemType,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    char *opt = kernelOpt->option;
    char ioMemName[128] = "";
    CHECK_STATUS(set_io_mem_name(inputMemType, outputMemType, ioMemName));
    std::string kernel = std::string("expand_") + ioMemName + std::to_string(nDims);
    UNI_STRCPY(kernelName, kernel.c_str());
    UNI_STRCPY(kernelOpt->sourceName, "expand");
    kernelOpt->kernelDataType = dt;
    CHECK_STATUS(set_value_define_opt(nDims, "DN", opt));
    CHECK_STATUS(set_io_mem_define_opt(inputMemType, outputMemType, opt));
    return SUCCESS;
}
#endif
