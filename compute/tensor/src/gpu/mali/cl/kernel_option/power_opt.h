#ifndef POWER_OPT
#define POWER_OPT

#include "common_opt.h"

inline EE set_power_opt_mali(bool useNchwFormat,
    DataType idt,
    DataType odt,
    GCLMemType inputMemType,
    GCLMemType outputMemType,
    PowerParamSpec p,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    kernelOpt->kernelDataType = DT_F16;
    std::string source = "power";
    UNI_STRCPY(kernelOpt->sourceName, source.c_str());
    char *opt = kernelOpt->option;
    std::string idtName = gcl_get_type(idt);
    std::string odtName = gcl_get_type(odt);
    CHECK_STATUS(add_macro(opt, "IT1", idtName));
    CHECK_STATUS(add_macro(opt, "IT4", idtName + std::string("4")));
    CHECK_STATUS(add_macro(opt, "OT1", odtName));
    CHECK_STATUS(add_macro(opt, "OT4", odtName + std::string("4")));
    CHECK_STATUS(add_macro_format(opt, useNchwFormat));
    CHECK_STATUS(add_macro_io(opt, inputMemType, outputMemType));
    if (p.power != 1) {
        CHECK_STATUS(add_macro(opt, "HAS_POW"));
    }
    std::string name = get_kernel_name(source, kernelOpt->option);
    CHECK_STATUS(add_macro(opt, "KERNEL_NAME", name.c_str()));
    UNI_STRCPY(kernelName, name.c_str());
    return SUCCESS;
}
#endif
