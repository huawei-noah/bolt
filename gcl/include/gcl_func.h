// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.



#ifndef H_GCL_FUNC
#define H_GCL_FUNC
#include <iostream>
#include "gcl_common.h"
#include "platform.h"
#include "context.h"
#include "program.h"
#include "memory.h"
#include "kernel.h"
#include "event.h"
#include "gcl_kernel_binmap.h"
#include <tuple>
#ifdef __cplusplus
extern "C" {
#endif
    inline EE gcl_get_device_name(GCLHandle_t handle) {
        cl_device_id device = handle->devices[handle->deviceId];
        U32 len;
        I8* data;
        CHECK_STATUS(get_device_info(device, CL_DEVICE_NAME, (void**)&data, &len));
        I8 devName[64];
        for(U32 i = 0; i < len - 1; i++) {
            if(data[i] == '-') {
                data[i] = '_';
            }
            devName[i] = data[i];
        }
        U32 version_len;
        free(data);
        CHECK_STATUS(get_device_info(device, CL_DEVICE_VERSION, (void**)&data, &version_len));
        std::string deviceV = std::string(data);
        U32 be = deviceV.find("r");
        U32 end = deviceV.find("p", be + 1);
        std::string numV = deviceV.substr(be + 1, end - be - 1);
        U32 i = atoi(numV.c_str());
        if(i >= 14) {
            devName[len - 1] = 'p';
            devName[len] = '\0';
        } else {
            devName[len - 1] = '\0';
        }
        free(data);
        handle->deviceBinmapName = devName;
        return SUCCESS;
    }

    inline EE gcl_create_handle(GCLHandle_t* handlePtr) {
   
       if(handlePtr == NULL) {
           printf("the handlePtr set to gcl_create_handle is NULL\n");
           return NULL_POINTER;
       }
       GCLHandle_t handle= new GCLHandle();
       handle->platformId = 0;
       handle->deviceId   = 0;
       handle->deviceType = CL_DEVICE_TYPE_GPU;
       handle->t_execute  = 0;
       U32 platformId = handle->platformId;
       U32 deviceId   = handle->deviceId;
       CHECK_STATUS(get_platforms(&handle->numPlatform, &handle->platforms));
       CHECK_STATUS(platform_get_devices(handle->platforms[platformId],
                                         handle->deviceType,
                                         &handle->numDevice,
                                         &handle->devices));
       CHECK_STATUS(create_context(handle->platforms[platformId],
                                   handle->numDevice,
                                   handle->devices,
                                   &handle->context));
       cl_queue_properties props[]={CL_QUEUE_PROPERTIES, 0, 0};
#ifdef _DEBUG                                               
       handle->queueProperties = CL_QUEUE_PROFILING_ENABLE;
       handle->eventPtr        = &handle->eventObj;
       props[1] = props[1] | CL_QUEUE_PROFILING_ENABLE;
#endif 
       CHECK_STATUS(create_command_queue_properties(handle->context,
                                                    handle->devices[deviceId],
                                                    props,
                                                    &handle->queue));
       CHECK_STATUS(gcl_get_device_name(handle));
       gcl_kernel_binmap_factory::instance()->create_gcl_kernel_binmap(handle->deviceBinmapName);
       gcl_kernel_binmap* kernel_binmap;
       U32 EE = gcl_kernel_binmap_container::instance()->get(handle->deviceBinmapName, &kernel_binmap);
       if(EE == NULL_POINTER) {
           std::cout << "warning: the kernel binmap is not found" << std::endl;
       } else {
           handle->binMapPtr = &kernel_binmap->binMap();
       }
       *handlePtr = handle;
       return SUCCESS;
    }
    
    inline void gcl_destroy_handle(GCLHandle_t handle) {
        U32 deviceId   = handle->deviceId;
        auto kernelMap = handle->kernelMap;
        auto kernelVec = handle->kernelVec;
        for(auto k : kernelMap) CHECK_STATUS(release_kernel(k.second));
        for(auto k : kernelVec) CHECK_STATUS(release_kernel(k.kernel));
        kernelMap.clear();
        kernelVec.clear();
        CHECK_STATUS(finish(handle->queue));
        CHECK_STATUS(release_command_queue(handle->queue));  
        CHECK_STATUS(release_context(handle->context)); 
        CHECK_STATUS(release_device(handle->devices[deviceId])); 
        free(handle->devices);
        free(handle->platforms);
        delete handle;
    }

    inline GCLMemDesc gcl_mem_desc(U32 stride[], U32 offset[], DataType dt, DataFormat memFormat){
        GCLMemDesc desc;
        U32 s0, s1, s2;
        s0 = stride[0];
        s1 = stride[1];
        s2 = stride[2];
        desc.stride[0] = s0;
        desc.stride[1] = s1;
        desc.stride[2] = s2;
        desc.offset[0] = offset[0];
        desc.offset[1] = offset[1];
        desc.offset[2] = offset[2];
        desc.memFormat = memFormat;
        desc.memType   = GCL_MEM_BUF;
        desc.num       = s0 * s1 * s2;
        desc.byteSize  = s0 * s1 * s2 * bytesOf(dt); 
        desc.flags     = CL_MEM_READ_WRITE;
        desc.host_ptr  = NULL;
        desc.imgFormat.image_channel_order     = CL_RGBA;
        desc.imgFormat.image_channel_data_type = CL_HALF_FLOAT;
        desc.use_map   = false;
        desc.map_ptr   = NULL;
        desc.has_alloc = false;
        return desc;
    }

    inline GCLMem_t gcl_create_gclmem(){
        GCLMem_t ret = new GCLMem;
        ret->mem = nullptr;
        U32 str[3] = {0, 0, 0};
        U32 off[3] = {0, 0, 0};
        ret->desc = gcl_mem_desc(str, off, DT_U8, DF_NCWHC4);
        return ret;
    }
    
    inline void gcl_destroy_gclmem(GCLMem_t mem){
        delete mem;
    }

    inline EE gcl_finish(GCLHandle_t handle) {
        CHECK_STATUS(finish(handle->queue));
        return SUCCESS;
    }

    inline EE gcl_release_memory(GCLMem_t gclMem) {
        if(gclMem->mem) {
            CHECK_STATUS(release_memory(gclMem->mem));
            gclMem->mem = NULL;
            gclMem->desc.has_alloc = false;
        }
        return SUCCESS;
    }

    inline EE gcl_unmap_memory(GCLHandle_t handle, GCLMem_t gclMem)
    {
        DEBUG_info("DATAUNMAP >>> enqueue_unmap_memory runInfo: ");           
        CHECK_STATUS(enqueue_unmap_memory(handle->queue, gclMem->mem, gclMem->desc.map_ptr, 
                                                      handle->numWaitEvents, handle->waitEvents, handle->eventPtr));
#ifdef _DEBUG                                                             
        double executeTime = 0;
        CHECK_STATUS(event_counting_time(handle->eventPtr, NULL, NULL, NULL, NULL, &executeTime));
        std::cout << "executeTime = " << executeTime << " us" << std::endl;
        CHECK_STATUS(gcl_finish(handle));
#endif               
        return SUCCESS;

    }


    inline EE gcl_produce_program_kernel_with_source(GCLHandle_t handle, 
                                                     U32*        len,
                                                     CI8*        src,
                                                     CI8*        option,
                                                     Program*    program,
                                                     U32         numKernel,
                                                     Kernel*     kernels) {
       U32 deviceId   = handle->deviceId;
       CHECK_STATUS(create_build_program_from_source(handle->context, len, src, handle->devices[deviceId], option, program));
       CHECK_STATUS(create_kernels_in_program(*program, numKernel, kernels));
       return SUCCESS;
    }

    inline EE gcl_get_program_info(Program     program,
                                   U8**        binary,
                                   U32*        len) {
        CHECK_STATUS(get_program_binary(program, binary, len));
        return SUCCESS;
    }

    inline EE gcl_kernelmap_put(GCLHandle_t     handle,
                                std::string     kernelName,
                                Kernel          kernel) {
        auto kernelMap = handle->kernelMap;
        kernelMap.insert(std::pair<std::string, Kernel>(kernelName, kernel));
        return SUCCESS;
    }

    inline Kernel gcl_kernelmap_get(GCLHandle_t     handle, 		                  
                                    std::string     kernelName) {
        auto kernelMap = handle->kernelMap;
        auto it = kernelMap.find(std::string(kernelName));
        if(it == kernelMap.end()) CHECK_STATUS(NOT_MATCH);
        return it->second;
    }
 
    inline EE gcl_create_kernel_binary(GCLHandle_t handle,
                                       CI8*        kernelname,
                                       Kernel*     kernel) {
                                       
        std::string binmapname = handle->deviceBinmapName;
        std::string binmap_kernelname = binmapname + "_" + std::string(kernelname);
        auto binMapPtr = handle->binMapPtr;
        auto it = binMapPtr->find(binmap_kernelname);
        if(it == binMapPtr->end()) {
            printf("get kernel %s failed\n", kernelname);
            return NULL_POINTER;
        }
        
        U32 length = it->second.len;
        CU8* data  = it->second.data;
        I32 binsta;
        Program program;
        CI8* options = "";
        Device device = handle->devices[handle->deviceId];
        CHECK_STATUS(create_program_from_binary(handle->context, device, &length, &data, &binsta, &program));
        CHECK_STATUS(build_program(program, device, options));
        CHECK_STATUS(create_kernel(program, kernelname, kernel));
        CHECK_STATUS(release_program(program));
        return SUCCESS;
    }

    inline EE gcl_get_kernel_from_map(GCLHandle_t handle,
                                      CI8*        kernelname,
                                      Kernel*     kernel) {
        std::string binmapname = handle->deviceBinmapName;
        std::string binmap_kernelname = binmapname + "_" + std::string(kernelname);
        auto kernelMap = handle->kernelMap;
        if(kernelMap.find(binmap_kernelname) == kernelMap.end()) {
            CHECK_STATUS(gcl_create_kernel_binary(handle, kernelname, kernel))
            CHECK_STATUS(gcl_kernelmap_put(handle, binmap_kernelname, *kernel));
        } else {
            *kernel = gcl_kernelmap_get(handle, binmap_kernelname);
        }
        return SUCCESS;
    }
                                         
 
    inline EE gcl_set_kernelVec(GCLHandle_t handle,
                                Kernel      kernel,
                                U32         work_dim,
                                U32         global_work_size[],
                                U32         local_work_size[]) {
        GCLKernelInfo kernelInfo;
        kernelInfo.kernel = kernel;
        kernelInfo.dim    = work_dim;
        switch(work_dim) {
            case 1:{
                kernelInfo.gs[0] = global_work_size[0];
                kernelInfo.ls[0] = local_work_size[0];
                break;}
            case 2:{
                kernelInfo.gs[0] = global_work_size[0];
                kernelInfo.gs[1] = global_work_size[1];
                kernelInfo.ls[0] = local_work_size[0];
                kernelInfo.ls[1] = local_work_size[1];
                break;}
            case 3:{
                kernelInfo.gs[0] = global_work_size[0];
                kernelInfo.gs[1] = global_work_size[1];
                kernelInfo.gs[2] = global_work_size[2];
                kernelInfo.ls[0] = local_work_size[0];
                kernelInfo.ls[1] = local_work_size[1];
                kernelInfo.ls[2] = local_work_size[2];
                break;}
            default:
                return NOT_SUPPORTED;
        }
        handle->kernelVec.push_back(kernelInfo);
        return SUCCESS;
    }
 
    inline EE gcl_run_kernelVec(GCLHandle_t handle) {
        U32 len            = handle->kernelVec.size();
        CommandQueue queue = handle->queue;
        U32 numWaitEvents  = handle->numWaitEvents;
        Event* waitEvents  = handle->waitEvents;
        Event* eventPtr    = handle->eventPtr;
        for(U32 i = 0 ; i < len; ++i) {
            auto kernelInfo = handle->kernelVec[i];
            CHECK_STATUS(enqueue_ndrange_kernel(queue, kernelInfo.kernel, kernelInfo.dim, 
                                                            NULL,          kernelInfo.gs, kernelInfo.ls, 
                                                            numWaitEvents, waitEvents,    eventPtr));
        }
        return SUCCESS;
    }

    inline EE gcl_run_kernel(GCLHandle_t handle, Kernel kernel, U32 work_dim, U32* gs, U32* ls, CI8* kernelName = NULL) {
        CHECK_STATUS(enqueue_ndrange_kernel(handle->queue, kernel, work_dim, 
                                                        NULL,          gs,     ls, 
                                                        handle->numWaitEvents, handle->waitEvents, handle->eventPtr));
#ifdef _DEBUG                                 
        std::string name = "unknown kernel";
        if(kernelName) name = kernelName;
        std::cout << "KERNEL>>>" <<name.c_str() << " runInfo: "<< std::endl;
        double executeTime = 0;
        CHECK_STATUS(event_counting_time(handle->eventPtr, NULL, NULL, NULL, NULL, &executeTime));
        std::cout << "executeTime = " << executeTime << " us"<<std::endl;
        CHECK_STATUS(gcl_finish(handle));
#else
        UNUSED(kernelName);
#endif            
        return SUCCESS;
    }

#ifdef _DEBUG                                 
    inline EE gcl_run_kernel_profiling(GCLHandle_t handle, Kernel kernel, U32 work_dim, U32* gs, U32* ls, CI8* kernelName = NULL) {
        std::string name = "unknown kernel";
        if(kernelName) name = kernelName;
        std::cout << "KERNEL>>>" <<name.c_str() << " runInfo: "<< std::endl;
        double totalTime = 0;
        double executeTime = 0;
        U32 loop = 10;
        for(U32 i = 0; i < loop; i++) {
            double t;
            CHECK_STATUS(enqueue_ndrange_kernel(handle->queue, kernel, work_dim, 
                                                            NULL,          gs,     ls, 
                                                            handle->numWaitEvents, handle->waitEvents, handle->eventPtr));
            CHECK_STATUS(event_counting_time(handle->eventPtr, NULL, NULL, NULL, NULL, &t));
            std::cout << "loop "<< i << " executeTime = " << t << " us" << std::endl;
            totalTime += t;
            
        }
        executeTime = totalTime / loop;
        std::cout << "executeTime = " << executeTime << " us for " << loop << " times average"<<std::endl;
        CHECK_STATUS(gcl_finish(handle));
        return SUCCESS;
    }
#endif            

    inline EE gcl_create_memory(GCLHandle_t handle, GCLMem_t gclMem) {
        GCLMemDesc_t desc = &gclMem->desc;
        if(!desc->has_alloc){
            switch(desc->memType) {
                case GCL_MEM_BUF:{
                    CHECK_STATUS(create_buffer(handle->context, desc->flags, desc->byteSize, desc->host_ptr, &gclMem->mem));
                    desc->has_alloc = true;
                    break;
                }
                case GCL_MEM_IMG_1D:{
                    CHECK_STATUS(create_image1D(handle->context, desc->flags, &desc->imgFormat, desc->stride[0], 0, desc->host_ptr, &gclMem->mem));
                    desc->has_alloc = true;
                    break;
                }
                case GCL_MEM_IMG_2D:{
                    CHECK_STATUS(create_image2D(handle->context, desc->flags, &desc->imgFormat, desc->stride[0], desc->stride[1], 0, desc->host_ptr, &gclMem->mem));
                    desc->has_alloc = true;
                    break;
                }
                case GCL_MEM_IMG_3D:{
                    CHECK_STATUS(create_image3D(handle->context, desc->flags, &desc->imgFormat, desc->stride[0], desc->stride[1], desc->stride[2], 0, 0, desc->host_ptr, &gclMem->mem));
                    desc->has_alloc = true;
                    break;
                }
                default: return NOT_SUPPORTED;
            }
        } else {
            //std::cout << "warning try to alloc the same gpu mem again without release" << std::endl;
        }
        return SUCCESS;
    }

    inline EE gcl_trans_memory(GCLHandle_t handle, void* src, void* dst, U32* size, GCLMemTransType type, cl_bool blocking)
    {
       DEBUG_info("DATATRANS>>>");
       switch(type) {
           case HOST_TO_DEVICE_BUF:{
               U8* hostPtr     = (U8*)src;
               GCLMem_t gclMem = (GCLMem_t)dst;
               CHECK_STATUS(enqueue_write_buffer(handle->queue, gclMem->mem, blocking, 0, *size, hostPtr, 
                                                             handle->numWaitEvents, handle->waitEvents, handle->eventPtr));
               DEBUG_info("enqueue_write_buffer runInfo: ");     
               break;
           }
           case HOST_TO_DEVICE_IMG:{
               U8* hostPtr     = (U8*)src;
               GCLMem_t gclMem = (GCLMem_t)dst;
               U32 origin[3] = {0, 0, 0};
               CHECK_STATUS(enqueue_write_image(handle->queue, gclMem->mem, blocking, origin, size, 0, 0, hostPtr,
                                                            handle->numWaitEvents, handle->waitEvents, handle->eventPtr));
               DEBUG_info("enqueue_write_image runInfo: ");
               break;
           }
           case DEVICE_BUF_TO_HOST:{
               U8* hostPtr     = (U8*)dst;
               GCLMem_t gclMem = (GCLMem_t)src;
               CHECK_STATUS(enqueue_read_buffer(handle->queue, gclMem->mem, blocking, 0, *size, hostPtr, 
                                                            handle->numWaitEvents, handle->waitEvents, handle->eventPtr));
               DEBUG_info("enqueue_read_buffer runInfo: ");
               break;
           }
           case DEVICE_IMG_TO_HOST:{
               U8* hostPtr     = (U8*)dst;
               GCLMem_t gclMem = (GCLMem_t)src;
               U32 origin[3] = {0, 0, 0};
               CHECK_STATUS(enqueue_read_image(handle->queue, gclMem->mem, blocking, origin, size, 0, 0, hostPtr,
                                                           handle->numWaitEvents, handle->waitEvents, handle->eventPtr));
               DEBUG_info("enqueue_read_image runInfo: ");
               break;
           }
           case DEVICE_BUF_TO_BUF:{
               GCLMem_t srcBuf = (GCLMem_t)src;
               GCLMem_t dstBuf = (GCLMem_t)dst;
               CHECK_STATUS(enqueue_copy_buffer(handle->queue, srcBuf->mem, dstBuf->mem, 0, 0, *size,
                                                            handle->numWaitEvents, handle->waitEvents, handle->eventPtr));
               DEBUG_info("enqueue_copy_buffer runInfo: ");
               break;
           }
           case DEVICE_BUF_TO_IMG:{
               return NOT_SUPPORTED;
               break;
           }
           case DEVICE_IMG_TO_BUF:{
               return NOT_SUPPORTED;
               break;
           }
           case DEVICE_IMG_TO_IMG:{
               return NOT_SUPPORTED;
               break;
           }
           default: return NOT_SUPPORTED;
       }
#ifdef _DEBUG                                                             
           double executeTime = 0;
           CHECK_STATUS(event_counting_time(handle->eventPtr, NULL, NULL, NULL, NULL, &executeTime));
           std::cout << "executeTime = " << executeTime << " us"<<std::endl;
           CHECK_STATUS(gcl_finish(handle));
#endif               
       return SUCCESS; 
    }

    inline EE gcl_map_memory(GCLHandle_t handle, GCLMem_t gclMem, U32* size, cl_map_flags flags, cl_bool blocking)
    {
        DEBUG_info("DATAMAP>>>");
        GCLMemDesc_t desc = &gclMem->desc;
        if (gclMem->desc.memType == GCL_MEM_BUF) {
            DEBUG_info("enqueue_map_buffer runInfo: ");
            CHECK_STATUS(enqueue_map_buffer(handle->queue, gclMem->mem, blocking, flags, 0, *size, 
                                                        handle->numWaitEvents, handle->waitEvents, handle->eventPtr, (void**)&desc->map_ptr));
        } else {
            return NOT_SUPPORTED;
        }
#ifdef _DEBUG                                                             
           double executeTime = 0;
           CHECK_STATUS(event_counting_time(handle->eventPtr, NULL, NULL, NULL, NULL, &executeTime));
           std::cout << "executeTime = " << executeTime << " us"<<std::endl;
           CHECK_STATUS(gcl_finish(handle));
#endif               
        return SUCCESS;
    }


    inline EE gcl_fill_memory_zero(GCLHandle_t handle, GCLMem_t gclMem) {
        if(gclMem->desc.memType == GCL_MEM_BUF) {
            U8 pat_val = 0;
            CHECK_STATUS(enqueue_fill_buffer(handle->queue, gclMem->mem, &pat_val, sizeof(pat_val), 0, gclMem->desc.byteSize,
                                                         handle->numWaitEvents, handle->waitEvents, handle->eventPtr));
            DEBUG_info("FILLMEM>>> enqueue_fill_buffer runInfo: ");
        } else {
            F32 color[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            U32 origin[3] = {0, 0, 0};
            U32 region[3];
            region[0] = gclMem->desc.stride[0];
            region[1] = gclMem->desc.stride[1];
            region[2] = gclMem->desc.stride[2];
            CHECK_STATUS(enqueue_fill_image(handle->queue, gclMem->mem, color, origin, region,
                                                        handle->numWaitEvents, handle->waitEvents, handle->eventPtr));
            DEBUG_info("FILLMEM>>> enqueue_fill_image runInfo: ");
        }
#ifdef _DEBUG        
        double executeTime = 0;
        CHECK_STATUS(event_counting_time(handle->eventPtr, NULL, NULL, NULL, NULL, &executeTime));
        std::cout << "executeTime = " << executeTime << " us"<<std::endl;
        CHECK_STATUS(gcl_finish(handle));
#endif        
        return SUCCESS;
    }

    inline EE gcl_get_mem_size(GCLMem_t gclMem, U32* size) {
        CHECK_STATUS(get_memory_size(gclMem->mem, size));
        return SUCCESS;
    }

    inline EE gcl_create_sub_buffer(U32 size, U32* offset, GCLMem_t src, cl_mem* subbuf){
        CHECK_STATUS(create_sub_buffer(src->mem, CL_MEM_READ_WRITE, *offset, size, subbuf));
        *offset += (size + 1023) / 1024 * 1024;
        return SUCCESS;
    }
 #ifdef __cplusplus
 }
 #endif
    template<typename Tuple, U32 N>
    struct DummpyWrapper{
        static void set_kernel_arg_wrapper(Kernel kernel, const Tuple& t) {
            DummpyWrapper<Tuple, N-1>::set_kernel_arg_wrapper(kernel, t);
            auto arg = std::get<N-1>(t);
            set_kernel_arg(kernel, N-1, sizeof(arg), (void*)&arg);
        }
    };
 
    template<typename Tuple>
    struct DummpyWrapper<Tuple, 0>{
        static void set_kernel_arg_wrapper(Kernel kernel, const Tuple& t) {
            UNUSED(kernel);
            UNUSED(t);
        }
    };
 
    template<typename ... Args>
    inline EE gcl_set_kernelArgs(Kernel kernel, Args ... args) {
        std::tuple<Args...> t = std::make_tuple(args...);
        DummpyWrapper<decltype(t), sizeof...(Args)>::set_kernel_arg_wrapper(kernel, t);
        return SUCCESS;
    }

#ifdef _DEBUG                                 
    template<typename T>
    inline EE gcl_print_memory(GCLHandle_t handle, GCLMem_t gclMem, CI8* gclMemName = NULL) {
        UNUSED(handle);
        UNUSED(gclMem);
        UNUSED(gclMemName);/*
        GCLMemDesc_t desc = &gclMem->desc;
        if(gclMemName) std::cout << "MEMORY>>>"<< gclMemName << " info:"<<std::endl;
        else std::cout << "unknown gclMem: " << std::endl;
        gcl_finish(handle);
        U8* hostPtr = nullptr;
        U32 s0 = desc->stride[0];
        U32 s1 = desc->stride[1];
        U32 s2 = desc->stride[2];
        switch(desc->memType) {
            case GCL_MEM_BUF:{
                U32 size = desc->byteSize;
                hostPtr = new U8[(size_t)size];
                gcl_trans_memory(handle, (void*)gclMem, (void*)hostPtr, &size, DEVICE_BUF_TO_HOST, CL_TRUE);
                break;
            }
            case GCL_MEM_IMG_1D:{
                U32 dim[3];
                dim[0] = s0;
                dim[1] = s1;
                dim[2] = s2;
                U32 size = desc->byteSize;
                hostPtr = new U8[(size_t)size];
                gcl_trans_memory(handle, (void*)gclMem, (void*)hostPtr, dim, DEVICE_IMG_TO_HOST, CL_TRUE);
                s0 = s0 * 4;
                break;
            }
            case GCL_MEM_IMG_2D:{
                break;
            }
            case GCL_MEM_IMG_3D:{
                break;
            }
            default: return NOT_SUPPORTED;
        }
        
        T* data = (T*)hostPtr;
        if(desc->memFormat == DF_NCHW) {
            std::cout << "Format: NCHW" << std::endl;
            std::cout << "s0 = " << s0 << std::endl;
            std::cout << "s1 = " << s1 << std::endl;
            std::cout << "s2 = " << s2 << std::endl;
            U32 num = 0;
            for(U32 i = 0; i < s2; i++) {
                U32 ii = i * s1 * s0;
                for(U32 j = 0; j < s1; j++) {
                    U32 jj = j * s0;
                    for(U32 k = 0; k < s0; k++) {
                        std::cout << 0.0 + data[ii + jj + k] << " ";
                        if(num >= 63) {std::cout << std::endl; goto end;}
                        num++;
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            }
        }

        if(desc->memFormat == DF_NCWHC4) {
            std::cout << "Format: NCWHC4" << std::endl;
            std::cout << "s0 * 4 = " << s0 * 4 << std::endl;
            std::cout << "s1     = " << s1 << std::endl;
            std::cout << "s2     = " << s2 << std::endl;
            U32 num = 0;
            for(U32 i = 0; i < s2; i++) {
                U32 ii = i * s1 * s0 * 4;
                for(U32 j = 0; j < s1; j++) {
                    U32 jj = j * s0 * 4;
                    for(U32 k = 0; k < s0 * 4; k++) {
                        std::cout << 0.0 + data[ii + jj + k] << " ";
                        if(num >= 63) {std::cout << std::endl; goto end;}
                        num++;
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            }
        }

        if(desc->memFormat == DF_NHWC) {
            std::cout << "Format: NHWC" << std::endl;
            std::cout << "s0     = " << s0 << std::endl;
            std::cout << "s1     = " << s1 << std::endl;
            std::cout << "s2     = " << s2 << std::endl;
            U32 num = 0;
            for(U32 i = 0; i < s2; i++) {
                U32 ii = i * s1 * s0;
                for(U32 j = 0; j < s1; j++) {
                    U32 jj = j * s0;
                    for(U32 k = 0; k < s0; k++) {
                        std::cout << 0.0 + data[ii + jj + k] << " ";
                        if(num >= 63) {std::cout << std::endl; goto end;}
                        num++;
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            }
        }

        if(desc->memFormat == DF_NCHWN4C4) {
            std::cout << "Format: NCHWN4C4" << std::endl;
            std::cout << "s0 * 16 = " << s0 * 16 << std::endl;
            std::cout << "s1      = " << s1 << std::endl;
            std::cout << "s2      = " << s2 << std::endl;
            U32 num = 0;
            for(U32 i = 0; i < s2; i++) {
                U32 ii = i * s1 * s0 * 16;
                for(U32 j = 0; j < s1; j++) {
                    U32 jj = j * s0 * 16;
                    for(U32 k = 0; k < s0 * 16; k++) {
                        std::cout << 0.0 + data[ii + jj + k] << " ";
                        if(num >= 63) {std::cout << std::endl; goto end;}
                        num++;
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            }
        }

        if(desc->memFormat == DF_NCWHN4C4) {
            std::cout << "Format: NCWHN4C4" << std::endl;
            std::cout << "s0 * 16 = " << s0 * 16 << std::endl;
            std::cout << "s1      = " << s1 << std::endl;
            std::cout << "s2      = " << s2 << std::endl;
            U32 num = 0;
            for(U32 i = 0; i < s2; i++) {
                U32 ii = i * s1 * s0 * 16;
                for(U32 j = 0; j < s1; j++) {
                    U32 jj = j * s0 * 16;
                    for(U32 k = 0; k < s0 * 16; k++) {
                        std::cout << 0.0 + data[ii + jj + k] << " ";
                        if(num >= 63) {std::cout << std::endl; goto end;}
                        num++;
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            }
        }

        if(desc->memFormat == DF_NCHWN4C4) {
            std::cout << "Format: NCHWN4C4" << std::endl;
            std::cout << "s0 * 4 = " << s0 * 4 << std::endl;
            std::cout << "s1     = " << s1 << std::endl;
            std::cout << "s2     = " << s2 << std::endl;
            U32 num = 0;
            for(U32 i = 0; i < s2; i++) {
                U32 ii = i * s1 * s0 * 4;
                for(U32 j = 0; j < s1; j++) {
                    U32 jj = j * s0 * 4;
                    for(U32 k = 0; k < s0 * 4; k++) {
                        std::cout << 0.0 + data[ii + jj + k] << " ";
                        if(num >= 63) {std::cout << std::endl; goto end;}
                        num++;
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            }
        }
        if(desc->memFormat == DF_NHWCN4) {
            std::cout << "Format: NHWCN4" << std::endl;
            std::cout << "s0 * 4 = " << s0 * 4 << std::endl;
            std::cout << "s1     = " << s1 << std::endl;
            std::cout << "s2     = " << s2 << std::endl;
            U32 num = 0;
            for(U32 i = 0; i < s2; i++) {
                U32 ii = i * s1 * s0 * 4;
                for(U32 j = 0; j < s1; j++) {
                    U32 jj = j * s0 * 4;
                    for(U32 k = 0; k < s0 * 4; k++) {
                        std::cout << 0.0 + data[ii + jj + k] << " ";
                        if(num >= 63) {std::cout << std::endl; goto end;}
                        num++;
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            }
        }       
end:        
        delete[] hostPtr;*/
        return SUCCESS;
    }

    template<typename T>
    inline EE gcl_print_buffer(GCLHandle_t handle, cl_mem mem, U32 num, CI8* bufferName = NULL) {
        UNUSED(handle);
        UNUSED(mem);
        UNUSED(num);
        UNUSED(bufferName);/*
        if(bufferName) std::cout << "BUFFER>>> "<< bufferName << " info:"<<std::endl;
        else std::cout << "Buffer>>> unknown info: " << std::endl;
        std::cout << "Element number = " << num << std::endl;
        U8* hostPtr = new U8[(size_t)num * sizeof(T)];
        CHECK_STATUS(enqueue_read_buffer(handle->queue, mem, CL_TRUE, 0, num * sizeof(T), (void*)hostPtr, 
            handle->numWaitEvents, handle->waitEvents, handle->eventPtr));
        T* val = (T*)hostPtr;
        for(U32 i = 0; i < num; i++){
            std::cout << val[i] << " ";
            if(i >= 63) break;
        }
        std::cout << std::endl;
        delete[] hostPtr;*/
        return SUCCESS;
    }
#endif    
#endif
