#ifndef TRANSPOSE_OPT
#define TRANSPOSE_OPT
#include "common_opt.h"
inline EE set_transpose_opt_mali(U32 nDims,
    DataType dt,
    GCLMemType inputMemType,
    GCLMemType outputMemType,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    char *opt = kernelOpt->option;
    char ioMemName[128] = "";
    CHECK_STATUS(set_io_mem_name(inputMemType, outputMemType, ioMemName));
    kernelOpt->kernelDataType = dt;
    std::string kernel = std::string("transpose_nchw_") + ioMemName + std::to_string(nDims);
    UNI_STRCPY(kernelName, kernel.c_str());
    UNI_STRCPY(kernelOpt->sourceName, "transpose_nchw");
    CHECK_STATUS(set_value_define_opt(nDims, "DN", opt));
    CHECK_STATUS(set_io_mem_define_opt(inputMemType, outputMemType, opt));
    return SUCCESS;
}
#endif
