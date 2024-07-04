#ifndef RESIZE_OPT
#define RESIZE_OPT

#include "common_opt.h"

inline EE set_resize_opt_mali(ResizeParamSpec p,
    DataFormat df,
    DataType idt,
    DataType odt,
    GCLMemType inputMemType,
    GCLMemType outputMemType,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    DataType dt = idt;
    if (bytesOf(odt) > bytesOf(idt)) {
        dt = odt;
    }
    if (bytesOf(dt) < 2) {
        dt = DT_F32;
    }
    kernelOpt->kernelDataType = dt;
    char *opt = kernelOpt->option;
    std::string source;
    if (p.mode == RESIZE_NEAREST) {
        source = "resize_nearest";
    } else if (p.mode == RESIZE_LINEAR) {
        source = "resize_bilinear";
    } else {
        UNI_ERROR_LOG("GPU currently not support this mode resize.\n");
        return NOT_SUPPORTED;
    }
    switch (p.trans_mode) {
        case COORDINATE_TRANS_HALF_PIXEL: {
            CHECK_STATUS(add_macro(opt, "USE_HALF_PIXEL"));
            break;
        }
        case COORDINATE_TRANS_PYTORCH_HALF_PIXEL: {
            CHECK_STATUS(add_macro(opt, "USE_PYTORCH_HALF_PIXEL"));
            break;
        }
        case COORDINATE_TRANS_ALIGN_CORNERS: {
            CHECK_STATUS(add_macro(opt, "USE_ALIGN_CORNERS"));
            break;
        }
        case COORDINATE_TRANS_ASYMMETRIC: {
            CHECK_STATUS(add_macro(opt, "USE_ASYMMETRIC"));
            break;
        }
        default:
            UNI_ERROR_LOG("GPU currently not support this trans mode in resize.\n");
            return NOT_SUPPORTED;
    }
    if (df == DF_NCHWC4) {
        CHECK_STATUS(add_macro(opt, "USE_NCHWC4"));
    } else if (df == DF_NHWC) {
        CHECK_STATUS(add_macro(opt, "USE_NHWC"));
    } else {
        CHECK_STATUS(add_macro(opt, "USE_NCHW"));
    }
    if (odt == DT_U8) {
        CHECK_STATUS(add_macro(opt, "OUTPUT_UCHAR"));
    }
    std::string idtName = gcl_get_type(idt);
    std::string odtName = gcl_get_type(odt);
    CHECK_STATUS(add_macro(opt, "IT", idtName));
    CHECK_STATUS(add_macro(opt, "IT4", idtName + "4"));
    CHECK_STATUS(add_macro(opt, "OT", odtName));
    CHECK_STATUS(add_macro_type(opt, kernelOpt->kernelDataType));
    CHECK_STATUS(add_macro_io(opt, inputMemType, outputMemType));
    UNI_STRCPY(kernelOpt->sourceName, source.c_str());
    std::string name = get_kernel_name(source, kernelOpt->option);
    CHECK_STATUS(add_macro(opt, "KERNEL_NAME", name.c_str()));
    UNI_STRCPY(kernelName, name.c_str());
    return SUCCESS;
}
#endif
