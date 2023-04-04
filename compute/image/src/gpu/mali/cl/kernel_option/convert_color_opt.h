#ifndef _H_CONVERT_COLOR_OPT
#define _H_CONVERT_COLOR_OPT

#include "common_opt.h"

inline EE set_convert_color_opt_mali(
    ConvertColorParamSpec p, DataType idt, DataType odt, char *kernelName, KernelOpt *kernelOpt)
{
#ifdef _USE_FP16
    kernelOpt->kernelDataType = DT_F16;
#else
    kernelOpt->kernelDataType = DT_F32;
#endif
    char *opt = kernelOpt->option;
    std::string idtName = gcl_get_type(idt);
    std::string odtName = gcl_get_type(odt);
    CHECK_STATUS(add_macro(opt, "IT", idtName));
    CHECK_STATUS(add_macro(opt, "OT", odtName));
    static std::map<ColorSpace, std::string> m = {
        {RGB_0_1, "USE_RGB_0_1"},
        {RGB_0_255, "USE_RGB_0_255"},
        {BGR_0_1, "USE_BGR_0_1"},
        {BGR_0_255, "USE_BGR_0_255"},
        {RGBA_0_1, "USE_RGBA_0_1"},
        {RGBA_0_255, "USE_RGBA_0_255"},
        {BGRA_0_1, "USE_BGRA_0_1"},
        {BGRA_0_255, "USE_BGRA_0_255"},
    };
    std::string source;
    if (p.src == YUV_NV21) {
        if (m.find(p.dst) != m.end()) {
            source = "yuv_nv21_to_rgb";
            CHECK_STATUS(add_macro(opt, m[p.dst]));
        } else {
            return NOT_SUPPORTED;
        }
    } else if (m.find(p.src) != m.end()) {
        CHECK_STATUS(add_macro(opt, m[p.src]));
        if (p.dst == YUV_NV21) {
            source = "rgb_to_yuv_nv21";
        } else {
            return NOT_SUPPORTED;
        }
    } else {
        return NOT_SUPPORTED;
    }
    UNI_STRCPY(kernelOpt->sourceName, source.c_str());
    std::string name = get_kernel_name(source, kernelOpt->option);
    CHECK_STATUS(add_macro(opt, "KERNEL_NAME", name.c_str()));
    UNI_STRCPY(kernelName, name.c_str());
    return SUCCESS;
}
#endif
