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
       handle->eventPtr   = NULL;
       handle->numWaitEvents = 0;
       handle->waitEvents = NULL;
       handle->t_execute  = 0;
       handle->t_total    = 0;
       handle->curOpName  = "unknow";
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
       *handlePtr = handle;
       return SUCCESS;
    }
    
    inline EE gcl_create_handle_profiling(GCLHandle_t* handlePtr) {
   
       if(handlePtr == NULL) {
           printf("the handlePtr set to gcl_create_handle is NULL\n");
           return NULL_POINTER;
       }
       GCLHandle_t handle= new GCLHandle();
       handle->platformId = 0;
       handle->deviceId   = 0;
       handle->deviceType = CL_DEVICE_TYPE_GPU;
       handle->eventPtr   = NULL;
       handle->numWaitEvents = 0;
       handle->t_execute  = 0;
       handle->t_total    = 0;
       handle->curOpName  = "unknow";
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
       handle->queueProperties = CL_QUEUE_PROFILING_ENABLE;
       handle->eventPtr        = &handle->eventObj;
       props[1] = props[1] | CL_QUEUE_PROFILING_ENABLE;
       CHECK_STATUS(create_command_queue_properties(handle->context,
                                                    handle->devices[deviceId],
                                                    props,
                                                    &handle->queue));
       CHECK_STATUS(gcl_get_device_name(handle));
       *handlePtr = handle;
       return SUCCESS;
    }

    inline void gcl_destroy_handle(GCLHandle_t handle) {
        U32 deviceId   = handle->deviceId;
        CHECK_STATUS(finish(handle->queue));
        for(auto k : handle->kernelMap) CHECK_STATUS(release_kernel(k.second));
        for(auto k : handle->kernelVec) CHECK_STATUS(release_kernel(k.kernel));
        handle->kernelMap.clear();
        handle->kernelVec.clear();
        CHECK_STATUS(release_command_queue(handle->queue));
        CHECK_STATUS(release_context(handle->context));
        CHECK_STATUS(release_device(handle->devices[deviceId])); 
        free(handle->devices);
        free(handle->platforms);
        delete handle;
    }

    inline EE gcl_create_queue_profiling(GCLHandle_t handle) {
   
       cl_queue_properties props[]={CL_QUEUE_PROPERTIES, 0, 0};
       handle->eventPtr        = &handle->eventObj;
       props[1] = props[1] | CL_QUEUE_PROFILING_ENABLE;
       CHECK_STATUS(create_command_queue_properties(handle->context,
                                                    handle->devices[handle->deviceId],
                                                    props,
                                                    &handle->queue_profiling));
       return SUCCESS;
    }

    inline EE gcl_destroy_queue_profiling(GCLHandle_t handle) {
        CHECK_STATUS(finish(handle->queue_profiling));
        CHECK_STATUS(release_command_queue(handle->queue_profiling));
        handle->eventPtr = NULL;
        return SUCCESS;
    }

    inline EE gcl_regist_binMap(GCLHandle_t handle){
       gcl_kernel_binmap_factory::instance()->create_gcl_kernel_binmap(handle->deviceBinmapName);
       gcl_kernel_binmap* kernel_binmap;
       U32 EE = gcl_kernel_binmap_container::instance()->get(handle->deviceBinmapName, &kernel_binmap);
       if(EE == NULL_POINTER) {
           DEBUG_info("warning: the kernel binmap is not found");
       } else {
           handle->binMapPtr = &kernel_binmap->binMap();
       }
       return SUCCESS;
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
        ret->mem = NULL;
        U32 str[3] = {0, 0, 0};
        U32 off[3] = {0, 0, 0};
        ret->desc = gcl_mem_desc(str, off, DT_U8, DF_NCWHC4);
        return ret;
    }

    inline EE gcl_release_memory(GCLMem_t gclMem) {
        if(gclMem->mem) {
            if(gclMem->subMem.size()) {
                for(auto p: gclMem->subMem) CHECK_STATUS(release_memory(p));
                gclMem->subMem.clear();
            }
            CHECK_STATUS(release_memory(gclMem->mem));
            gclMem->mem = NULL;
            gclMem->desc.has_alloc = false;
        }
        return SUCCESS;
    }
    
    inline void gcl_destroy_gclmem(GCLMem_t mem){
        CHECK_STATUS(gcl_release_memory(mem));
        delete mem;
    }

    inline EE gcl_finish(GCLHandle_t handle) {
        CHECK_STATUS(finish(handle->queue));
        return SUCCESS;
    }


    inline EE gcl_unmap_memory(GCLHandle_t handle, GCLMem_t gclMem)
    {
        for(auto p : gclMem->mapPtrArray) {
            CHECK_STATUS(enqueue_unmap_memory(handle->queue, gclMem->mem, (void*)p, 
                                    handle->numWaitEvents, handle->waitEvents, handle->eventPtr));
#ifdef _DEBUG                                                             
            DEBUG_info_s("DATAUNMAP>>> enqueue_unmap_memory runInfo:");
            double executeTime = 0;
            CHECK_STATUS(event_counting_time(handle->eventPtr, NULL, NULL, NULL, NULL, &executeTime));
            CHECK_STATUS(release_event(handle->eventObj));
            DEBUG_info("executeTime = " << executeTime << " us"); 
            CHECK_STATUS(gcl_finish(handle));
#endif               
        }
        gclMem->mapPtrArray.clear();
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
        handle->kernelMap.insert(std::pair<std::string, Kernel>(kernelName, kernel));
        return SUCCESS;
    }

    inline Kernel gcl_kernelmap_get(GCLHandle_t     handle, 		                  
                                    std::string     kernelName) {
        auto it = handle->kernelMap.find(std::string(kernelName));
        if(it == handle->kernelMap.end()) CHECK_STATUS(NOT_MATCH);
        return it->second;
    }
 
    inline EE gcl_create_kernel_binary(GCLHandle_t handle,
                                       CI8*        kernelName,
                                       Kernel*     kernel) {
                                       
        std::string binmapname = handle->deviceBinmapName;
        std::string binmap_kernelname = binmapname + "_" + std::string(kernelName);
        auto binMapPtr = handle->binMapPtr;
        auto it = binMapPtr->find(binmap_kernelname);
        if(it == binMapPtr->end()) {
            DEBUG_info("get kernel " << kernelName << " failed");
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
        CHECK_STATUS(create_kernel(program, kernelName, kernel));
        CHECK_STATUS(release_program(program));
        return SUCCESS;
    }

    inline EE gcl_get_kernel_from_map(GCLHandle_t handle,
                                      CI8*        kernelName,
                                      Kernel*     kernel) {
        std::string binmapname = handle->deviceBinmapName;
        std::string binmap_kernelname = binmapname + "_" + std::string(kernelName);
        if(handle->kernelMap.find(binmap_kernelname) == handle->kernelMap.end()) {
            CHECK_STATUS(gcl_create_kernel_binary(handle, kernelName, kernel))
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
                                U32         local_work_size[],
                                CI8*        kernelName = NULL) {
        GCLKernelInfo kernelInfo;
        kernelInfo.kernel = kernel;
        kernelInfo.dim    = work_dim;
        kernelInfo.name   = handle->curOpName + "_" + std::string(kernelName);
        switch(work_dim) {
            case 1: {
                kernelInfo.gs[0] = global_work_size[0];
                kernelInfo.gs[1] = 1;
                kernelInfo.gs[2] = 1;
                kernelInfo.ls[0] = local_work_size[0];
                kernelInfo.ls[1] = 0;
                kernelInfo.ls[2] = 0;
                break;}
            case 2: {
                kernelInfo.gs[0] = global_work_size[0];
                kernelInfo.gs[1] = global_work_size[1];
                kernelInfo.gs[2] = 1;
                kernelInfo.ls[0] = local_work_size[0];
                kernelInfo.ls[1] = local_work_size[1];
                kernelInfo.ls[2] = 0;
                break;}
            case 3: {
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
            CHECK_STATUS(enqueue_ndrange_kernel(queue, kernelInfo.kernel,  kernelInfo.dim, NULL, 
                kernelInfo.gs, kernelInfo.ls, numWaitEvents, waitEvents, eventPtr));
#ifdef _DEBUG                                 
            DEBUG_info_s("KERNEL>>> " << kernelInfo.name << " runInfo:");
            double executeTime = 0;
            CHECK_STATUS(event_counting_time(eventPtr, NULL, NULL, NULL, NULL, &executeTime));
            CHECK_STATUS(release_event(*eventPtr));
            handle->t_execute = executeTime;
            DEBUG_info("executeTime = " << executeTime << " us");
            CHECK_STATUS(gcl_finish(handle));
#endif            
        }
        return SUCCESS;
    }

    inline EE gcl_run_kernelVec_timing(GCLHandle_t handle, U32 be, U32 end, std::vector<double> *kernelArrayTime = NULL) {
        if(handle->queueProperties & CL_QUEUE_PROFILING_ENABLE) {
            double executeTime = 0;
            double totalTime = 0;
            CommandQueue queue = handle->queue;
            U32 numWaitEvents  = handle->numWaitEvents;
            Event* waitEvents  = handle->waitEvents;
            Event* eventPtr    = handle->eventPtr;
            for(U32 i = be ; i < end; ++i) {
                auto kernelInfo = handle->kernelVec[i];
                CHECK_STATUS(enqueue_ndrange_kernel(queue, kernelInfo.kernel,  kernelInfo.dim, NULL, 
                                        kernelInfo.gs, kernelInfo.ls, numWaitEvents, waitEvents, eventPtr));
                CHECK_STATUS(event_counting_time(handle->eventPtr, NULL, NULL, NULL, NULL, &executeTime));
                CHECK_STATUS(release_event(handle->eventObj));
                totalTime += executeTime;
                if(kernelArrayTime) (*kernelArrayTime).push_back(executeTime);
            }
            handle->t_execute = totalTime;
            return SUCCESS;
        } 
        return NOT_SUPPORTED;
    }

    inline EE gcl_clean_kernelVec(GCLHandle_t handle) {
        for(auto k : handle->kernelVec) CHECK_STATUS(release_kernel(k.kernel));
        handle->kernelVec.clear();
        return SUCCESS;
    }

    inline EE gcl_run_kernel(GCLHandle_t handle, Kernel kernel, U32 work_dim, U32* gs, U32* ls, CI8* kernelName = NULL) {
#ifdef _DEBUG    
        std::string name = "unknown kernel";
        if(kernelName) name = handle->curOpName + "_" + std::string(kernelName);
        DEBUG_info_s("KERNEL>>> " << name.c_str() << " runInfo:");
#endif        
        CHECK_STATUS(enqueue_ndrange_kernel(handle->queue, kernel, work_dim, 
                                                        NULL,          gs,     ls, 
                                                        handle->numWaitEvents, handle->waitEvents, handle->eventPtr));
#ifdef _DEBUG                                 
        double executeTime = 0;
        CHECK_STATUS(event_counting_time(handle->eventPtr, NULL, NULL, NULL, NULL, &executeTime));
        CHECK_STATUS(release_event(handle->eventObj));
        handle->t_execute = executeTime;
        DEBUG_info("executeTime = " << executeTime << " us");
        CHECK_STATUS(gcl_finish(handle));
#else
        UNUSED(kernelName);
#endif            
        return SUCCESS;
    }

    inline U32 get_next_ls_size(U32 ls_size) {
        return (ls_size << 1);
    }
    inline EE gcl_run_kernel_select_ls(GCLHandle_t handle, GCLKernelInfo* kernelInfo) {
        auto kernel = kernelInfo->kernel;
        auto work_dim = kernelInfo->dim;
        auto gs = kernelInfo->gs;
        double minTime = DBL_MAX;
        double time;
        U32 test_ls[3];
        U32 best_ls[3];
        U32 test_gs[3];
        U32 maxSize = 384;
        U32 gs_x = 256;
        U32 gs_y = (work_dim > 1) ? 256 : 1;
        U32 gs_z = (work_dim > 2) ? gs[2] : 1;
        for(U32 z = 1; z <= gs_z; z = get_next_ls_size(z)) {
            if(0 != gs_z % z) continue;
            for(U32 y = 1; y <= gs_y; y = get_next_ls_size(y)) {
                if(0 != gs_y % y) continue;
                for(U32 x = 1; x <= gs_x; x = get_next_ls_size(x)) {
                    if(0 != gs_x % x) continue;
                    U32 total = x * y * z;
                    if(total <= maxSize) {
                       test_gs[0] = (gs[0] + x - 1) / x * x;
                       test_gs[1] = (gs[1] + y - 1) / y * y;
                       test_gs[2] = (gs[2] + z - 1) / z * z;
                       test_ls[0] = x;
                       test_ls[1] = y;
                       test_ls[2] = z;
                       CHECK_STATUS(enqueue_ndrange_kernel(handle->queue_profiling, kernel, work_dim, NULL, test_gs, test_ls, 
                        handle->numWaitEvents, handle->waitEvents, handle->eventPtr));
                        CHECK_STATUS(event_counting_time(handle->eventPtr, NULL, NULL, NULL, NULL, &time));
                        if(minTime > time){
                            minTime = time;
                            best_ls[0] = test_ls[0];
                            best_ls[1] = test_ls[1];
                            best_ls[2] = test_ls[2];
                        }
                        CHECK_STATUS(release_event(handle->eventObj));
                    }
                }
            }
        }
        test_ls[0] = 0;
        test_ls[1] = 0;
        test_ls[2] = 0;
        CHECK_STATUS(enqueue_ndrange_kernel(handle->queue_profiling, kernel, work_dim, NULL, gs, test_ls, handle->numWaitEvents, handle->waitEvents, handle->eventPtr));
        CHECK_STATUS(event_counting_time(handle->eventPtr, NULL, NULL, NULL, NULL, &time));
        if(minTime > time){
            minTime = time;
            best_ls[0] = test_ls[0];
            best_ls[1] = test_ls[1];
            best_ls[2] = test_ls[2];
        }
        CHECK_STATUS(release_event(handle->eventObj));
        if(best_ls[0] != 0 && best_ls[1] != 0 && best_ls[2] != 0) {
            kernelInfo->gs[0] = (gs[0] + best_ls[0] - 1) / best_ls[0] * best_ls[0];
            kernelInfo->gs[1] = (gs[1] + best_ls[1] - 1) / best_ls[1] * best_ls[1];
            kernelInfo->gs[2] = (gs[2] + best_ls[2] - 1) / best_ls[2] * best_ls[2];
        }
        kernelInfo->ls[0] = best_ls[0];
        kernelInfo->ls[1] = best_ls[1];
        kernelInfo->ls[2] = best_ls[2];
        handle->t_execute = minTime;
#ifdef _DEBUG          
        DEBUG_info_s("SELECT LS KERNEL>>> " << kernelInfo->name.c_str() << " runInfo:");
        DEBUG_info_s("best ls = " << best_ls[0] << " " << best_ls[1] << " " << best_ls[2] << " ");
        DEBUG_info(" executeTime = " << minTime << " us");
#endif
        return SUCCESS;
    }

    inline EE gcl_run_kernelVec_select_ls(GCLHandle_t handle, std::vector<U32> kernelIndex) {
        if(kernelIndex.size() == 0) return SUCCESS;
        CHECK_STATUS(gcl_create_queue_profiling(handle));
        for(auto index : kernelIndex) {
            auto kernelInfo = handle->kernelVec[index];
            CHECK_STATUS(gcl_run_kernel_select_ls(handle, &kernelInfo));
            handle->kernelVec[index].gs[0] = kernelInfo.gs[0];
            handle->kernelVec[index].gs[1] = kernelInfo.gs[1];
            handle->kernelVec[index].gs[2] = kernelInfo.gs[2];
            handle->kernelVec[index].ls[0] = kernelInfo.ls[0];
            handle->kernelVec[index].ls[1] = kernelInfo.ls[1];
            handle->kernelVec[index].ls[2] = kernelInfo.ls[2];
        }
        CHECK_STATUS(gcl_destroy_queue_profiling(handle));
        return SUCCESS;
    }
    
#ifdef _DEBUG                                 
    inline EE gcl_run_kernel_profiling(GCLHandle_t handle, Kernel kernel, U32 work_dim, U32* gs, U32* ls, CI8* kernelName = NULL) {
        std::string name = "unknown kernel";
        if(kernelName) name = kernelName;
        DEBUG_info_s("KERNEL>>> " << name.c_str() << " runInfo:");
        double totalTime = 0;
        double executeTime = 0;
        U32 loop = 10;
        for(U32 i = 0; i < loop; i++) {
            double t;
            CHECK_STATUS(enqueue_ndrange_kernel(handle->queue, kernel, work_dim, 
                                                            NULL,          gs,     ls, 
                                                            handle->numWaitEvents, handle->waitEvents, handle->eventPtr));
            CHECK_STATUS(event_counting_time(handle->eventPtr, NULL, NULL, NULL, NULL, &t));
            CHECK_STATUS(release_event(handle->eventObj));
            DEBUG_info("loop " << i << " executeTime = " << t << " us");
            totalTime += t;
        }
        executeTime = totalTime / loop;
        DEBUG_info("executeTime = " << executeTime << " us for " << loop << " times average");
        CHECK_STATUS(gcl_finish(handle));
        return SUCCESS;
    }
#endif            

    inline EE gcl_create_memory(GCLHandle_t handle, GCLMem_t gclMem) {
        GCLMemDesc_t desc = &gclMem->desc;
        if(!desc->has_alloc){
            switch(desc->memType) {
                case GCL_MEM_BUF: {
                    CHECK_STATUS(create_buffer(handle->context, desc->flags, desc->byteSize, desc->host_ptr, &gclMem->mem));
                    desc->has_alloc = true;
                    break;
                }
                case GCL_MEM_IMG_1D: {
                    CHECK_STATUS(create_image1D(handle->context, desc->flags, &desc->imgFormat, desc->stride[0], 0, desc->host_ptr, &gclMem->mem));
                    desc->has_alloc = true;
                    break;
                }
                case GCL_MEM_IMG_2D: {
                    CHECK_STATUS(create_image2D(handle->context, desc->flags, &desc->imgFormat, desc->stride[0], desc->stride[1], 0, desc->host_ptr, &gclMem->mem));
                    desc->has_alloc = true;
                    break;
                }
                case GCL_MEM_IMG_3D: {
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

    inline EE gcl_trans_memory(GCLHandle_t handle, void* src, void* dst, U32* size, GCLMemTransType type, cl_bool blocking, U32* offset = NULL)
    {
        DEBUG_info_s("DATATRANS>>>");
        switch(type) {
            case HOST_TO_DEVICE_BUF: {
                U8* hostPtr     = (U8*)src;
                GCLMem_t gclMem = (GCLMem_t)dst;
                U32 dstOff = (offset) ? offset[0] : 0;
                CHECK_STATUS(enqueue_write_buffer(handle->queue, gclMem->mem, blocking, dstOff, *size, hostPtr, 
                    handle->numWaitEvents, handle->waitEvents, handle->eventPtr));
                DEBUG_info_s("enqueue_write_buffer runInfo: ");     
                break;
            }
            case HOST_TO_DEVICE_IMG: {
                U8* hostPtr     = (U8*)src;
                GCLMem_t gclMem = (GCLMem_t)dst;
                U32 origin[3] = {0, 0, 0};
                if(offset) {
                    origin[0] = offset[0];
                    origin[1] = offset[1];
                    origin[2] = offset[2];
                }
                CHECK_STATUS(enqueue_write_image(handle->queue, gclMem->mem, blocking, origin, size, 0, 0, hostPtr,
                    handle->numWaitEvents, handle->waitEvents, handle->eventPtr));
                DEBUG_info_s("enqueue_write_image runInfo: ");
                break;
            }
            case DEVICE_BUF_TO_HOST: {
                U8* hostPtr     = (U8*)dst;
                GCLMem_t gclMem = (GCLMem_t)src;
                U32 srcOff = (offset) ? offset[0] : 0;
                CHECK_STATUS(enqueue_read_buffer(handle->queue, gclMem->mem, blocking, srcOff, *size, hostPtr, 
                    handle->numWaitEvents, handle->waitEvents, handle->eventPtr));
                DEBUG_info_s("enqueue_read_buffer runInfo: ");
                break;
            }
            case DEVICE_IMG_TO_HOST: {
                U8* hostPtr     = (U8*)dst;
                GCLMem_t gclMem = (GCLMem_t)src;
                U32 origin[3] = {0, 0, 0};
                if(offset) {
                    origin[0] = offset[0];
                    origin[1] = offset[1];
                    origin[2] = offset[2];
                }
                CHECK_STATUS(enqueue_read_image(handle->queue, gclMem->mem, blocking, origin, size, 0, 0, hostPtr,
                    handle->numWaitEvents, handle->waitEvents, handle->eventPtr));
                DEBUG_info_s("enqueue_read_image runInfo: ");
                break;
            }
            case DEVICE_BUF_TO_BUF: {
                GCLMem_t srcBuf = (GCLMem_t)src;
                GCLMem_t dstBuf = (GCLMem_t)dst;
                U32 srcOff = 0;
                U32 dstOff = 0;
                if(offset) {
                    srcOff = offset[0];
                    dstOff = offset[1];
                }
                CHECK_STATUS(enqueue_copy_buffer(handle->queue, srcBuf->mem, dstBuf->mem, srcOff, dstOff, *size,
                    handle->numWaitEvents, handle->waitEvents, handle->eventPtr));
                DEBUG_info_s("enqueue_copy_buffer runInfo: ");
                break;
            }
            case DEVICE_BUF_TO_IMG: {
                GCLMem_t srcBuf = (GCLMem_t)src;
                GCLMem_t dstImg = (GCLMem_t)dst;
                U32 origin[3] = {0, 0, 0};
                U32 srcOff = 0;
                if(offset) {
                    srcOff    = offset[0];
                    origin[0] = offset[1];
                    origin[1] = offset[2];
                    origin[2] = offset[3];
                }
                CHECK_STATUS(enqueue_copy_buffer_to_image(handle->queue, srcBuf->mem, dstImg->mem, srcOff, origin, size, 
                    handle->numWaitEvents, handle->waitEvents, handle->eventPtr))
                DEBUG_info_s("enqueue_copy_buffer_to_image runInfo: ");
                break;
            }
            case DEVICE_IMG_TO_BUF: {
                GCLMem_t srcImg = (GCLMem_t)src;
                GCLMem_t dstBuf = (GCLMem_t)dst;
                U32 origin[3] = {0, 0, 0};
                U32 dstOff = 0;
                if(offset) {
                    origin[0] = offset[0];
                    origin[1] = offset[1];
                    origin[2] = offset[2];
                    dstOff    = offset[3];
                }
                CHECK_STATUS(enqueue_copy_image_to_buffer(handle->queue, srcImg->mem, dstBuf->mem, origin, size, dstOff, 
                    handle->numWaitEvents, handle->waitEvents, handle->eventPtr))
                DEBUG_info_s("enqueue_copy_image_to_buffer runInfo: ");
                break;
            }
            case DEVICE_IMG_TO_IMG: {
                return NOT_SUPPORTED;
                break;
            }
            default: return NOT_SUPPORTED;
        }
#ifdef _DEBUG                                                             
        double executeTime = 0;
        CHECK_STATUS(event_counting_time(handle->eventPtr, NULL, NULL, NULL, NULL, &executeTime));
        CHECK_STATUS(release_event(handle->eventObj));
        DEBUG_info("executeTime = " << executeTime << " us");
        CHECK_STATUS(gcl_finish(handle));
#endif               
       return SUCCESS; 
    }

    inline EE gcl_trans_buffer_rect(GCLHandle_t handle, void* src, void* dst, U32* host_org, U32* buf_org, U32* region, U32 host_row_pitch, U32 host_slice_pitch,
        U32 buf_row_pitch, U32 buf_slice_pitch, GCLMemTransType type, cl_bool blocking) {
        DEBUG_info_s("DATATRANS>>>");
        switch(type) {
            case HOST_TO_DEVICE_BUF: {
                GCLMem_t dstBuf = (GCLMem_t)dst;
                CHECK_STATUS(enqueue_write_buffer_rect(handle->queue, dstBuf->mem, blocking, buf_org, host_org, region, buf_row_pitch, buf_slice_pitch, 
                    host_row_pitch, host_slice_pitch, src, handle->numWaitEvents, handle->waitEvents, handle->eventPtr));
                DEBUG_info_s("enqueue_write_buffer_rect runInfo: ");     
                break;
            }
            case DEVICE_BUF_TO_HOST: {
                return NOT_SUPPORTED;
                break;
            }
            default: return NOT_SUPPORTED;
        }
#ifdef _DEBUG                                                             
        double executeTime = 0;
        CHECK_STATUS(event_counting_time(handle->eventPtr, NULL, NULL, NULL, NULL, &executeTime));
        CHECK_STATUS(release_event(handle->eventObj));
        DEBUG_info("executeTime = " << executeTime << " us");
        CHECK_STATUS(gcl_finish(handle));
#endif               
       return SUCCESS; 
    }

    inline EE gcl_map_memory(GCLHandle_t handle, GCLMem_t gclMem, U32*offset, U32* size, cl_map_flags flags, cl_bool blocking)
    {
        DEBUG_info_s("DATAMAP>>> enqueue_map_buffer runInfo:");
        GCLMemDesc_t desc = &gclMem->desc;
        if (gclMem->desc.memType == GCL_MEM_BUF) {
            CHECK_STATUS(enqueue_map_buffer(handle->queue, gclMem->mem, blocking, flags, *offset, *size, 
                                                        handle->numWaitEvents, handle->waitEvents, handle->eventPtr, (void**)&desc->map_ptr));
            gclMem->mapPtrArray.push_back(desc->map_ptr);
        } else {
            return NOT_SUPPORTED;
        }
#ifdef _DEBUG                                                             
           double executeTime = 0;
           CHECK_STATUS(event_counting_time(handle->eventPtr, NULL, NULL, NULL, NULL, &executeTime));
           CHECK_STATUS(release_event(handle->eventObj));
           DEBUG_info("executeTime = " << executeTime << " us");
           CHECK_STATUS(gcl_finish(handle));
#endif               
        return SUCCESS;
    }


    inline EE gcl_fill_memory_zero(GCLHandle_t handle, GCLMem_t gclMem) {
        if(gclMem->desc.memType == GCL_MEM_BUF) {
            DEBUG_info_s("FILLMEM>>> enqueue_fill_buffer runInfo:");
            U8 pat_val = 0;
            CHECK_STATUS(enqueue_fill_buffer(handle->queue, gclMem->mem, &pat_val, sizeof(pat_val), 0, gclMem->desc.byteSize,
                                                         handle->numWaitEvents, handle->waitEvents, handle->eventPtr));
        } else {
            DEBUG_info_s("FILLMEM>>> enqueue_fill_image runInfo:");
            F32 color[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            U32 origin[3] = {0, 0, 0};
            U32 region[3];
            region[0] = gclMem->desc.stride[0];
            region[1] = gclMem->desc.stride[1];
            region[2] = gclMem->desc.stride[2];
            CHECK_STATUS(enqueue_fill_image(handle->queue, gclMem->mem, color, origin, region,
                                                        handle->numWaitEvents, handle->waitEvents, handle->eventPtr));
        }
#ifdef _DEBUG        
        double executeTime = 0;
        CHECK_STATUS(event_counting_time(handle->eventPtr, NULL, NULL, NULL, NULL, &executeTime));
        CHECK_STATUS(release_event(handle->eventObj));
        DEBUG_info("executeTime = " << executeTime << " us");
        CHECK_STATUS(gcl_finish(handle));
#endif        
        return SUCCESS;
    }

    inline EE gcl_get_mem_size(GCLMem_t gclMem, U32* size) {
        CHECK_STATUS(get_memory_size(gclMem->mem, size));
        return SUCCESS;
    }

    inline EE gcl_create_sub_buffer(U32 size, U32* offset, GCLMem_t src, Mem* subbuf){
        CHECK_STATUS(create_sub_buffer(src->mem, CL_MEM_READ_WRITE, *offset, size, subbuf));
        src->subMem.push_back(*subbuf);
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

    inline std::string gclMemDesc2Str(GCLMemDesc desc) {
        char buff[128];
        snprintf(buff, sizeof(buff), "memFormat: %d, ", desc.memFormat);
        std::string descStr = buff;
        descStr += "stride(";
        for(U32 i = 0; i < 3; i++) {
                descStr += std::to_string(desc.stride[i]);
                if(i < 2) descStr += ",";
        }
        descStr += "), ";
        descStr += "offset(";
        for(U32 i = 0; i < 3; i++) {
                descStr += std::to_string(desc.offset[i]);
                if(i < 2) descStr += ",";
        }
        descStr += ")";
        return descStr;
    }
#ifdef _DEBUG                                 
    template<typename T>
    inline EE gcl_print_memory(GCLHandle_t handle, GCLMem_t gclMem, CI8* gclMemName = NULL) {
        UNUSED(handle);
        UNUSED(gclMem);
        UNUSED(gclMemName);
/*        GCLMemDesc_t desc = &gclMem->desc;
        if(gclMemName) std::cout << "MEMORY>>>"<< gclMemName << " info:"<<std::endl;
        else std::cout << "unknown gclMem: " << std::endl;
        gcl_finish(handle);
        U8* hostPtr = nullptr;
        U32 s0 = desc->stride[0];
        U32 s1 = desc->stride[1];
        U32 s2 = desc->stride[2];
        switch(desc->memType) {
            case GCL_MEM_BUF: {
                U32 size = desc->byteSize;
                hostPtr = new U8[(size_t)size];
                gcl_trans_memory(handle, (void*)gclMem, (void*)hostPtr, &size, DEVICE_BUF_TO_HOST, CL_TRUE);
                break;
            }
            case GCL_MEM_IMG_1D: {
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
            case GCL_MEM_IMG_2D: {
                break;
            }
            case GCL_MEM_IMG_3D: {
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

        if(desc->memFormat == DF_NHWC || desc->memFormat == DF_HWCN) {
            if(desc->memFormat == DF_NHWC) std::cout << "Format: NHWC" << std::endl;
            if(desc->memFormat == DF_HWCN) std::cout << "Format: HWCN" << std::endl;
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
    inline EE gcl_print_buffer(GCLHandle_t handle, Mem mem, U32 num, CI8* bufferName = NULL) {
        UNUSED(handle);
        UNUSED(mem);
        UNUSED(num);
        UNUSED(bufferName);
/*        if(bufferName) std::cout << "BUFFER>>> "<< bufferName << " info:"<<std::endl;
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

    template<typename T>
    inline EE gcl_write_buf_to_bin(GCLHandle_t handle, Mem buf, U32 size, CI8* dataName) {
        U32 num = size / sizeof(T);
        U8*  hostPtr = new U8[size];
        F32* hostPtrTran = new F32[num];
        CHECK_STATUS(enqueue_read_buffer(handle->queue, buf, CL_TRUE, 0, size, hostPtr, 
                                handle->numWaitEvents, handle->waitEvents, handle->eventPtr));
        T* val = (T*)hostPtr;
        for(U32 i = 0; i < num; i++) hostPtrTran[i] = (F32)val[i];

        FILE* outfile;
        std::string fileName = dataName;
        replace(fileName.begin(), fileName.end(), '/', '_');
        replace(fileName.begin(), fileName.end(), '.', '_');
        replace(fileName.begin(), fileName.end(), ' ', '_');
        fileName += "_gpu";
        fileName +=".out";
        outfile = fopen(fileName.c_str(), "wb");
        if(outfile == NULL) {
            DEBUG_info("waring fopen outfile " << fileName <<" failed"); 
            delete[] hostPtr;
            delete[] hostPtrTran;
            return SUCCESS;
        }
        fwrite(hostPtrTran, sizeof(float), num, outfile);
        fclose(outfile);
        delete[] hostPtr;
        delete[] hostPtrTran;
        return SUCCESS;
    }
    template<typename T>
    inline EE gcl_write_data_to_bin(GCLHandle_t handle, TensorDesc tensorDesc, void* ptr, U32 ptrType, CI8* dataName = NULL) {
        /*ptrType:
         *GPU: 0
         *CPU: 1
         */
        DataFormat tdf;
        DataType   tdt;
        U32 tn, tc, th, tw;
        U32 dims;
        tn = 1; tc = 1; th = 1; tw = 1;
        dims = tensorDesc.nDims;
        switch(dims) {
            case 1: 
                tensor1dGet(tensorDesc, &tdt, &tw);
                break;
            case 2: 
                tensor2dfGet(tensorDesc, &tdt, &tdf, &th, &tw);
                break;
            case 3: 
                tensor3dGet(tensorDesc, &tdt, &tdf, &tc, &th, &tw);
                break;
            case 4: 
                tensor4dGet(tensorDesc, &tdt, &tdf, &tn, &tc, &th, &tw);
                break;
            default: CHECK_STATUS(NOT_SUPPORTED);
        }
        U32  num = tn * tc * th * tw;
        F32* hostPtrTran = new F32[num];

        if(ptrType == 0) {
            GCLMem_t mem    = (GCLMem_t)ptr;
            GCLMemDesc desc = mem->desc;
            GCLMemType type = desc.memType;
            DataFormat df   = desc.memFormat;
            U8* hostPtr = nullptr;
            U32 s0 = desc.stride[0];
            U32 s1 = desc.stride[1];
            U32 off0 = desc.offset[0];
            U32 off1 = desc.offset[1];
            U32 byteSize = desc.byteSize;
            hostPtr = new U8[(size_t)byteSize];

            GCLMemTransType tranType = DEVICE_BUF_TO_HOST;
            U32 size[3] = {byteSize, 1, 1};
            if(type == GCL_MEM_IMG_1D) {
                tranType = DEVICE_IMG_TO_HOST;
                size[0] = s0;
            }
            gcl_trans_memory(handle, (void*)mem, (void*)hostPtr, size, tranType, CL_TRUE);
            
            T* val = (T*) hostPtr;
            if(df == DF_NCWHC4) {
                if(tdf == DF_NCHW) {
                    for(U32 i = 0; i < num; i++) {
                        U32 iw = i % tw;
                        U32 ih = (i / tw) % th;
                        U32 ic = i / (tw * th);
                        hostPtrTran[i] = (float)(val[((ic / 4) * s1 + iw + off1) * s0 * 4 + (ih + off0) * 4 + (ic & 3)]);
                    } 
                }
                if(tdf == DF_MKT) {
                    for(U32 i = 0; i < num; i++) {
                        U32 ih = i % tw;
                        U32 ic = i / tw;
                        U32 in_off = ((ic / 4) * s1 + off1) * s0 * 4 + (ih + off0) * 4 + (ic & 3);
                        hostPtrTran[i] = (float)val[in_off];
                    }
                }
            } else if(df == DF_NCHW || df == DF_NHWC) {
                for(U32 i = 0; i < num; i++) {
                    U32 iw = i % tw;
                    U32 ih = (i / tw) % th;
                    U32 ic = i / (tw * th);
                    hostPtrTran[i] = (float)(val[(ic * s1 + ih + off1) * s0 + (iw + off0)]);
                }
            } else if(df == DF_NORMAL) {
                for(U32 i = 0; i < num; i++) hostPtrTran[i] = (float)val[i];
            } else {
                DEBUG_info("warning write GPU memory " << dataName <<" to bin, format not support: " << df);
                delete[] hostPtrTran;
                delete[] hostPtr;
                return SUCCESS;
            }

            delete[] hostPtr;
        }

        if(ptrType == 1) {
            T* val = (T*) ptr;
            if(tdf == DF_NCHWC8) {
                for(U32 i = 0; i < num; i++) {
                    U32 iw = i % tw;
                    U32 ih = (i / tw) % th;
                    U32 ic = i / (tw * th);
                    hostPtrTran[i] = (float)(val[((ic / 8) * th + ih) * tw * 8 + iw * 8 + (ic & 7)]);
                }
            } else if(tdf == DF_NORMAL || tdf == DF_NCHW) {
                for(U32 i = 0; i < num; i++) {
                    hostPtrTran[i] = (float)(val[i]);
                }
            } else if(tdf == DF_MTK) {
                for(U32 i = 0; i < num; i++) {
                    U32 it = i % th;
                    U32 ik = i / th;
                    U32 in_off = it * tw + ik;
                    hostPtrTran[i] = (float)(val[in_off]);//write as MKT, for compare with gpu
                }
            } else {
                DEBUG_info("warning write CPU memory" << dataName <<" to bin, format not support: " << tdf);
                delete[] hostPtrTran;
                return SUCCESS;
            }
        }
 
        FILE* outfile;
        std::string fileName = dataName;
        replace(fileName.begin(), fileName.end(), '/', '_');
        replace(fileName.begin(), fileName.end(), '.', '_');
        replace(fileName.begin(), fileName.end(), ' ', '_');
        if(ptrType == 0) fileName += "_gpu";
        if(ptrType == 1) fileName += "_cpu";
        fileName +=".out";
        
        outfile = fopen(fileName.c_str(), "wb");
        if(outfile == NULL) {
            DEBUG_info("waring fopen outfile " << fileName <<" failed"); 
            delete[] hostPtrTran;
            return SUCCESS;
        }
        fwrite(hostPtrTran, sizeof(float), num, outfile);
        fclose(outfile);
        delete[] hostPtrTran;
        return SUCCESS;
    }
#endif    
#endif
