#ifndef COMMON_OPT
#define COMMON_OPT
#include "gcl_kernel_type.h"
#include "error.h"
#include "ocl_context.h"

inline EE set_chars_define_opt(const char *optName, char *&opt)
{
    std::string sopt = "-D";
    sopt += optName;
    sopt += " ";
    strcpy(opt, sopt.c_str());
    opt += sopt.length();
    return SUCCESS;
}

inline EE set_value_define_opt(U32 val, const char *valName, char *&opt)
{
    std::string sopt = "-D ";
    std::string sval = std::to_string(val);
    sopt += valName;
    sopt += "=";
    sopt += sval;
    sopt += " ";
    strcpy(opt, sopt.c_str());
    opt += sopt.length();
    return SUCCESS;
}

inline EE set_activation_define_opt(ActivationMode activeMode, char *&opt)
{
    std::string sopt = " ";
    switch (activeMode) {
        case ACTIVATION_NULL:
            sopt = "-D AM= ";
            break;
        case ACTIVATION_RELU:
            sopt = "-DUSE_RELU -D AM=relu_ ";
            break;
        case ACTIVATION_RELU6:
            sopt = "-DUSE_RELU6 -D AM=relu6_ ";
            break;
        case ACTIVATION_H_SIGMOID:
            sopt = "-DUSE_HSIGMOID -D AM=hsigmoid_ ";
            break;
        case ACTIVATION_H_SWISH:
            sopt = "-DUSE_HSWISH -D AM=hswish_ ";
            break;
        case ACTIVATION_GELU:
            sopt = "-DUSE_GELU -D AM=gelu_ ";
            break;
        case ACTIVATION_TANH:
            sopt = "-DUSE_TANH -D AM=tanh_ ";
            break;
        case ACTIVATION_SIGMOID:
            sopt = "-DUSE_SIGMOID -D AM=sigmoid_ ";
            break;
        case ACTIVATION_ABS:
            sopt = "-DUSE_ABS -D AM=abs_ ";
            break;
        case ACTIVATION_LOG:
            sopt = "-DUSE_LOG -D AM=log_ ";
            break;
        case ACTIVATION_NEG:
            sopt = "-DUSE_NEG -D AM=neg_ ";
            break;
        default:
            CHECK_STATUS(NOT_SUPPORTED);
            break;
    }
    strcpy(opt, sopt.c_str());
    opt += sopt.length();
    return SUCCESS;
}

inline EE set_activation_mode_name(ActivationMode activeMode, char *name)
{
    std::string sname = "";
    switch (activeMode) {
        case ACTIVATION_NULL:
            break;
        case ACTIVATION_RELU:
            sname = "relu_";
            break;
        case ACTIVATION_RELU6:
            sname = "relu6_";
            break;
        case ACTIVATION_H_SIGMOID:
            sname = "hsigmoid_";
            break;
        case ACTIVATION_H_SWISH:
            sname = "hswish_";
            break;
        case ACTIVATION_GELU:
            sname = "gelu_";
            break;
        case ACTIVATION_TANH:
            sname = "tanh_";
            break;
        case ACTIVATION_SIGMOID:
            sname = "sigmoid_";
            break;
        case ACTIVATION_ABS:
            sname = "abs_";
            break;
        case ACTIVATION_LOG:
            sname = "log_";
            break;
        case ACTIVATION_NEG:
            sname = "neg_";
            break;
        default:
            CHECK_STATUS(NOT_SUPPORTED);
            break;
    }
    strcpy(name, sname.c_str());
    return SUCCESS;
}

inline EE set_eltwise_define_opt(EltwiseMode eltwiseMode, char *&opt)
{
    std::string sopt = " ";
    switch (eltwiseMode) {
        case ELTWISE_MAX:
            sopt = "-DUSE_MAX -D EM=max_ ";
            break;
        case ELTWISE_MIN:
            sopt = "-DUSE_MIN -D EM=min_ ";
            break;
        case ELTWISE_SUM:
            sopt = "-DUSE_SUM -D EM=sum_ ";
            break;
        case ELTWISE_SUB:
            sopt = "-DUSE_SUB -D EM=sub_ ";
            break;
        case ELTWISE_PROD:
            sopt = "-DUSE_PROD -D EM=prod_ ";
            break;
        case ELTWISE_DIV:
            sopt = "-DUSE_DIV -D EM=div_ ";
            break;
        default:
            CHECK_STATUS(NOT_SUPPORTED);
            break;
    }
    strcpy(opt, sopt.c_str());
    opt += sopt.length();
    return SUCCESS;
}

inline EE set_eltwise_mode_name(EltwiseMode eltwiseMode, char *name)
{
    std::string sname = "";
    switch (eltwiseMode) {
        case ELTWISE_MAX:
            sname = "max_";
            break;
        case ELTWISE_MIN:
            sname = "min_";
            break;
        case ELTWISE_SUM:
            sname = "sum_";
            break;
        case ELTWISE_SUB:
            sname = "sub_";
            break;
        case ELTWISE_PROD:
            sname = "prod_";
            break;
        case ELTWISE_DIV:
            sname = "div_";
            break;
        default:
            CHECK_STATUS(NOT_SUPPORTED);
            break;
    }
    strcpy(name, sname.c_str());
    return SUCCESS;
}

inline EE set_io_mem_define_opt(GCLMemType inputType, GCLMemType outputType, char *&opt)
{
    bool useInputImg = (inputType == GCL_MEM_BUF) ? false : true;
    bool useOutputImg = (outputType == GCL_MEM_BUF) ? false : true;
    std::string def = "";
    if (useInputImg) {
        def += "-DUSE_INPUT_IMG ";
    }
    if (useOutputImg) {
        def += "-DUSE_OUTPUT_IMG ";
    }
    if (useInputImg && !useOutputImg) {
        def += "-D IOM=im_ ";
    } else if (!useInputImg && useOutputImg) {
        def += "-D IOM=om_ ";
    } else if (useInputImg && useOutputImg) {
        def += "-D IOM=iom_ ";
    } else {
        def += "-D IOM= ";
    }
    strcpy(opt, def.c_str());
    opt += def.length();
    return SUCCESS;
}

inline EE set_io_mem_name(GCLMemType inputType, GCLMemType outputType, char *name)
{
    bool useInputImg = (inputType == GCL_MEM_BUF) ? false : true;
    bool useOutputImg = (outputType == GCL_MEM_BUF) ? false : true;
    std::string sname = "";
    if (useInputImg && !useOutputImg) {
        sname = "im_";
    } else if (!useInputImg && useOutputImg) {
        sname = "om_";
    } else if (useInputImg && useOutputImg) {
        sname = "iom_";
    }
    strcpy(name, sname.c_str());
    return SUCCESS;
}

inline EE set_io_mems_name_and_define_opts(GCLMemType *inputMemType,
    GCLMemType *outputMemType,
    U32 inputNum,
    U32 outputNum,
    char *name,
    char *&opt)
{
    std::string im = "";
    std::string om = "";
    std::string iomDef = "";
    bool hasInputImg = false;
    bool hasOutputImg = false;
    for (U32 i = 0; i < inputNum; i++) {
        iomDef = "USE_INPUT_IMG";
        if (inputMemType[i] == GCL_MEM_IMG_3D) {
            hasInputImg = true;
            if (inputNum > 1) {
                im += std::to_string(i);
                if (i > 0) {
                    iomDef += std::to_string(i);
                }
            }
            CHECK_STATUS(set_chars_define_opt(iomDef.c_str(), opt));
        }
    }
    if (hasInputImg) {
        im = "i" + im;
    }

    for (U32 i = 0; i < outputNum; i++) {
        iomDef = "USE_OUTPUT_IMG";
        if (outputMemType[i] == GCL_MEM_IMG_3D) {
            hasOutputImg = true;
            if (outputNum > 1) {
                om += std::to_string(i);
                if (i > 0) {
                    iomDef += std::to_string(i);
                }
            }
            CHECK_STATUS(set_chars_define_opt(iomDef.c_str(), opt));
        }
    }
    if (hasOutputImg) {
        om = "o" + om;
    }

    std::string iom = im + om;
    iomDef = " IOM=";
    if (iom != "") {
        iom += "m_";
        iomDef += iom;
    }
    CHECK_STATUS(set_chars_define_opt(iomDef.c_str(), opt));

    strcpy(name, iom.c_str());
    return SUCCESS;
}

inline EE set_data_type_name(DataType dt, char *name)
{
    std::string sname = "";
    if (dt == DT_F16) {
        sname = "_f16";
    } else if (dt == DT_F32) {
        sname = "_f32";
    } else if (dt == DT_I32) {
        sname = "_i32";
    } else if (dt == DT_U32) {
        sname = "_u32";
    } else {
        return NOT_SUPPORTED;
    }
    strcpy(name, sname.c_str());
    return SUCCESS;
}

inline EE set_data_type_define_opt(DataType dt, char *&opt)
{
    std::string sopt = "";
    if (dt == DT_F16) {
        sopt = "-D DT=_f16 ";
    } else if (dt == DT_F32) {
        sopt = "-D DT=_f32 ";
    } else if (dt == DT_I32) {
        sopt = "-D DT=_i32 ";
    } else if (dt == DT_U32) {
        sopt = "-D DT=_u32 ";
    } else {
        return NOT_SUPPORTED;
    }
    strcpy(opt, sopt.c_str());
    opt += sopt.length();
    return SUCCESS;
}

inline EE set_common_opt(DataType dt,
    GCLMemType inputMemType,
    GCLMemType outputMemType,
    const char *sourceName,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    char ioMemName[128] = "";
    CHECK_STATUS(set_io_mem_name(inputMemType, outputMemType, ioMemName));
    sprintf(kernelName, "%s_%s", sourceName, ioMemName);
    strcpy(kernelOpt->sourceName, sourceName);
    kernelOpt->kernelDataType = dt;
    char *opt = kernelOpt->option;
    CHECK_STATUS(set_io_mem_define_opt(inputMemType, outputMemType, opt));
    return SUCCESS;
}

inline bool check_qualcomm_device(char *devName = nullptr)
{
    bool useQualcommDev = OCLContext::getInstance().handle->useQualcommDev;
    if (devName) {
        std::string dev = "";
        if (useQualcommDev) {
            dev = "_qc";
        }
        strcpy(devName, dev.c_str());
    }
    return useQualcommDev;
}

inline EE add_qcom_acc_16_bit_opt(char *&opt)
{
    std::string qcom_acc = "-qcom-accelerate-16-bit ";
    strcpy(opt, qcom_acc.c_str());
    opt += qcom_acc.length();
    return SUCCESS;
}
#endif
