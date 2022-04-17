#ifndef FILL_MEMORY_ZERO_VEC4_OPT
#define FILL_MEMORY_ZERO_VEC4_OPT
#include "common_opt.h"
inline EE set_fill_memory_zero_vec4_opt_mali(
    DataType dt, GCLMemType outputMemType, char *kernelName, KernelOpt *kernelOpt)
{
    char *opt = kernelOpt->option;
    char ioMemName[128] = "";
    CHECK_STATUS(set_io_mem_name(GCL_MEM_BUF, outputMemType, ioMemName));
    kernelOpt->kernelDataType = dt;
    char dtName[128];
    CHECK_STATUS(set_data_type_name(dt, dtName));
    std::string buffer =
        std::string("fill_memory_zero_vec4_") + std::string(ioMemName) + std::string(dtName);
    UNI_STRCPY(kernelName, buffer.c_str());
    UNI_STRCPY(kernelOpt->sourceName, "fill_memory_zero_vec4");
    CHECK_STATUS(set_data_type_define_opt(dt, opt));
    CHECK_STATUS(set_io_mem_define_opt(GCL_MEM_BUF, outputMemType, opt));
    return SUCCESS;
}
#endif
