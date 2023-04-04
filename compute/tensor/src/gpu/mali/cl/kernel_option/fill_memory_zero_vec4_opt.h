#ifndef FILL_MEMORY_ZERO_VEC4_OPT
#define FILL_MEMORY_ZERO_VEC4_OPT

#include "common_opt.h"

inline EE set_fill_memory_zero_vec4_opt_mali(
    DataType dt, GCLMemType outputMemType, char *kernelName, KernelOpt *kernelOpt)
{
    kernelOpt->kernelDataType = dt;
    std::string source = "fill_memory_zero_vec4";
    UNI_STRCPY(kernelOpt->sourceName, source.c_str());
    char *opt = kernelOpt->option;
    CHECK_STATUS(add_macro_type(opt, dt));
    CHECK_STATUS(add_macro_io(opt, GCL_MEM_BUF, outputMemType));
    std::string name = get_kernel_name(source, kernelOpt->option);
    CHECK_STATUS(add_macro(opt, "KERNEL_NAME", name.c_str()));
    UNI_STRCPY(kernelName, name.c_str());
    return SUCCESS;
}
#endif
