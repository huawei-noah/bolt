#ifndef TILE_OPT
#define TILE_OPT
#include "common_opt.h"
inline EE set_tile_opt_mali(U32 nDims,
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
    sprintf(kernelName, "tile_%s%d", ioMemName, nDims);
    sprintf(kernelOpt->sourceName, "tile");
    CHECK_STATUS(set_value_define_opt(nDims, "DN", opt));
    CHECK_STATUS(set_io_mem_define_opt(inputMemType, outputMemType, opt));
    return SUCCESS;
}
#endif
