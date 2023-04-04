#ifndef _H_BILATERAL_SLICE_APPLY_OPT
#define _H_BILATERAL_SLICE_APPLY_OPT

#include "common_opt.h"

inline EE set_bilateral_slice_apply_pre_opt_mali(DataType idt,
    DataType gdt,
    BilateralSliceApplyMode mode,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    kernelOpt->kernelDataType = gdt;
    std::string source = "bilateral_slice_apply_pre";
    UNI_STRCPY(kernelOpt->sourceName, source.c_str());
    char *opt = kernelOpt->option;
    CHECK_STATUS(add_macro_type(opt, gdt));
    std::string name = get_kernel_name(source, kernelOpt->option);
    CHECK_STATUS(add_macro(opt, "KERNEL_NAME", name.c_str()));
    UNI_STRCPY(kernelName, name.c_str());
    return SUCCESS;
}

inline EE set_bilateral_slice_apply_c12_opt_mali(DataType idt,
    DataType gdt,
    BilateralSliceApplyMode mode,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    kernelOpt->kernelDataType = gdt;
    std::string source = "bilateral_slice_apply_c12";
    UNI_STRCPY(kernelOpt->sourceName, source.c_str());
    char *opt = kernelOpt->option;
    CHECK_STATUS(add_macro_type(opt, gdt));
    if (idt == DT_U8) {
        CHECK_STATUS(add_macro(opt, "UCHAR"));
    }
    if (mode == BILATERAL_SLICE_APPLY_CONV) {
        CHECK_STATUS(add_macro(opt, "CONV"));
    }
    std::string name = get_kernel_name(source, kernelOpt->option);
    CHECK_STATUS(add_macro(opt, "KERNEL_NAME", name.c_str()));
    UNI_STRCPY(kernelName, name.c_str());
    return SUCCESS;
}
#endif
