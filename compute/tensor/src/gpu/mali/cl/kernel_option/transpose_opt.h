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
    kernelOpt->kernelDataType = dt;
    std::string source = "transpose_nchw";
    UNI_STRCPY(kernelOpt->sourceName, source.c_str());
    char *opt = kernelOpt->option;
    CHECK_STATUS(add_macro_type(opt, dt));
    CHECK_STATUS(add_macro_io(opt, inputMemType, outputMemType));
    CHECK_STATUS(add_macro(opt, "DN", nDims));
    std::string name = get_kernel_name(source, kernelOpt->option);
    CHECK_STATUS(add_macro(opt, "KERNEL_NAME", name.c_str()));
    UNI_STRCPY(kernelName, name.c_str());
    return SUCCESS;
}
#endif
