#ifndef SOFTMAX_OPT
#define SOFTMAX_OPT
#include "common_opt.h"
inline EE set_softmax_opt_mali(U32 axis,
    bool useNchwFormat,
    DataType dt,
    GCLMemType inputMemType,
    GCLMemType outputMemType,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    char ioMemName[128] = "";
    CHECK_STATUS(set_io_mem_name(inputMemType, outputMemType, ioMemName));
    std::string formatName = "";
    if (useNchwFormat) {
        formatName = "nchw_";
    }
    std::string kernel = std::string("softmax_") + ioMemName + formatName + std::to_string(axis);
    UNI_STRCPY(kernelName, kernel.c_str());
    UNI_STRCPY(kernelOpt->sourceName, "softmax");
    kernelOpt->kernelDataType = dt;
    char *opt = kernelOpt->option;
    CHECK_STATUS(set_value_define_opt(axis, "AXIS", opt));
    if (useNchwFormat) {
        CHECK_STATUS(set_chars_define_opt("USE_NCHW", opt));
    }
    CHECK_STATUS(set_io_mem_define_opt(inputMemType, outputMemType, opt));
    return SUCCESS;
}

inline EE set_softmax_vec_reduce_opt_mali(bool useNchwFormat,
    DataType dt,
    GCLMemDesc inputMemDesc,
    GCLMemDesc outputMemDesc,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    char *opt = kernelOpt->option;
    char ioMemName[128] = "";
    GCLMemType inputMemType = inputMemDesc.memType;
    GCLMemType outputMemType = outputMemDesc.memType;
    CHECK_STATUS(set_io_mem_name(inputMemType, outputMemType, ioMemName));
    std::string formatName = "";
    if (useNchwFormat) {
        formatName = "nchw_";
    }
    std::string inputAxis = "";
    std::string outputAxis = "";
    if (inputMemType != GCL_MEM_BUF) {
        for (U32 i = 0; i < inputMemDesc.nDims; i++) {
            if (inputMemDesc.dims[i] != 1) {
                if (i == 0) {
                    inputAxis = "ix_";
                    CHECK_STATUS(set_chars_define_opt("INPUT_IMG_AXIS_X", opt));
                    break;
                } else if (i == 1) {
                    inputAxis = "iy_";
                    CHECK_STATUS(set_chars_define_opt("INPUT_IMG_AXIS_Y", opt));
                    break;
                } else {
                    inputAxis = "iz_";
                    CHECK_STATUS(set_chars_define_opt("INPUT_IMG_AXIS_Z", opt));
                    break;
                }
            }
        }
    }
    if (outputMemType != GCL_MEM_BUF) {
        for (U32 i = 0; i < outputMemDesc.nDims; i++) {
            if (outputMemDesc.dims[i] != 1) {
                if (i == 0) {
                    outputAxis = "ox_";
                    CHECK_STATUS(set_chars_define_opt("OUTPUT_IMG_AXIS_X", opt));
                    break;
                } else if (i == 1) {
                    outputAxis = "oy_";
                    CHECK_STATUS(set_chars_define_opt("OUTPUT_IMG_AXIS_Y", opt));
                    break;
                } else {
                    outputAxis = "oz_";
                    CHECK_STATUS(set_chars_define_opt("OUTPUT_IMG_AXIS_Z", opt));
                    break;
                }
            }
        }
    }
    std::string kernel =
        std::string("softmax_vec_reduce_") + ioMemName + formatName + inputAxis + outputAxis;
    UNI_STRCPY(kernelName, kernel.c_str());
    UNI_STRCPY(kernelOpt->sourceName, "softmax_vec_reduce");
    kernelOpt->kernelDataType = dt;
    if (useNchwFormat) {
        CHECK_STATUS(set_chars_define_opt("USE_NCHW", opt));
    }
    CHECK_STATUS(set_io_mem_define_opt(inputMemType, outputMemType, opt));
    return SUCCESS;
}
#endif
