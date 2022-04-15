// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <tests/tools/TestTools.h>

#include <cstdio>
#include <tests/tools/TestTools.h>

#include <training/common/Common.h>
#include <training/common/Tensor.h>

namespace UT
{
using namespace std;
using namespace raul;

TEST(TestOpenCL, CheckOpenCLDevicesGpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    // cl platforms

    cl_uint num_platforms_available = 0;
    //char str_buffer[1024];
    EXPECT_EQ(CL_SUCCESS, clGetPlatformIDs(0, NULL, &num_platforms_available));

    cout << "Found " << num_platforms_available << " OpenCL platforms" << endl;

    vector<cl_platform_id> cl_platforms(num_platforms_available);

    EXPECT_EQ(CL_SUCCESS, clGetPlatformIDs(num_platforms_available, cl_platforms.data(), NULL));

    vector<pair<string, int>> params = {
        { "Name", CL_PLATFORM_NAME }, { "Vendor", CL_PLATFORM_VENDOR }, { "Version", CL_PLATFORM_VERSION }, { "Profile", CL_PLATFORM_PROFILE }, { "Extensions", CL_PLATFORM_EXTENSIONS }
    };

    vector<pair<string, int>> dparams = { { "Name", CL_DEVICE_NAME }, { "Version", CL_DEVICE_VERSION }, { "Driver", CL_DRIVER_VERSION }, { "C Version", CL_DEVICE_OPENCL_C_VERSION } };

    for (cl_uint i = 0; i < num_platforms_available; ++i)
    {
        cout << "Platform " << i << ". " << endl;
        for (const auto& p : params)
        {
            size_t param_size = 0;
            EXPECT_EQ(CL_SUCCESS, clGetPlatformInfo(cl_platforms[i], p.second, 0, NULL, &param_size));
            vector<char> str_buffer(param_size);
            EXPECT_EQ(CL_SUCCESS, clGetPlatformInfo(cl_platforms[i], p.second, param_size, str_buffer.data(), NULL));
            cout << "  " << p.first << ": " << string(str_buffer.begin(), str_buffer.end()) << endl;
        }
        cl_uint num_devices_available;
        EXPECT_EQ(CL_SUCCESS, clGetDeviceIDs(cl_platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices_available));
        vector<cl_device_id> cl_devices(num_devices_available);
        EXPECT_EQ(CL_SUCCESS, clGetDeviceIDs(cl_platforms[i], CL_DEVICE_TYPE_ALL, num_devices_available, cl_devices.data(), NULL));
        cout << endl;
        for (cl_uint j = 0; j < num_devices_available; j++)
        {
            cout << "  Device " << j << endl;
            for (const auto& p : dparams)
            {
                size_t param_size = 0;
                EXPECT_EQ(CL_SUCCESS, clGetDeviceInfo(cl_devices[j], p.second, 0, NULL, &param_size));
                vector<char> str_buffer(param_size);
                EXPECT_EQ(CL_SUCCESS, clGetDeviceInfo(cl_devices[j], p.second, param_size, str_buffer.data(), NULL));
                cout << "  " << p.first << ": " << string(str_buffer.begin(), str_buffer.end()) << endl;
            }

            // Get device max clock frequency
            cl_uint max_clock_freq;
            EXPECT_EQ(CL_SUCCESS, clGetDeviceInfo(cl_devices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(max_clock_freq), &max_clock_freq, NULL));
            cout << "    Max clock frequency: " << max_clock_freq << "MHz" << endl;
            // Get device max compute units available
            cl_uint max_compute_units_available;
            EXPECT_EQ(CL_SUCCESS, clGetDeviceInfo(cl_devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(max_compute_units_available), &max_compute_units_available, NULL));
            cout << "    Max compute units: " << max_clock_freq << endl;

            // Get device global mem size
            cl_ulong global_mem_size;
            EXPECT_EQ(CL_SUCCESS, clGetDeviceInfo(cl_devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem_size), &global_mem_size, NULL));
            cout << "    Global mem size: " << global_mem_size / (1024 * 1024) << "Mb" << endl;

            // Get device max compute units available
            cl_ulong max_mem_alloc_size;
            EXPECT_EQ(CL_SUCCESS, clGetDeviceInfo(cl_devices[j], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_mem_alloc_size), &max_mem_alloc_size, NULL));
            cout << "    Max mem alloc size: " << max_mem_alloc_size / (1024 * 1024) << "Mb" << endl;

            // Get device local mem size
            cl_ulong local_mem_size;
            EXPECT_EQ(CL_SUCCESS, clGetDeviceInfo(cl_devices[j], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem_size), &local_mem_size, NULL));
            cout << "    Local mem size: " << local_mem_size / 1024 << "Kb" << endl;

            // Get device max work group size
            size_t max_work_group_size;
            EXPECT_EQ(CL_SUCCESS, clGetDeviceInfo(cl_devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL));
            cout << "    Max workgroup size: " << max_work_group_size << endl;

            // Get device max work item dim
            cl_uint max_work_item_dims;
            EXPECT_EQ(CL_SUCCESS, clGetDeviceInfo(cl_devices[j], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(max_work_item_dims), &max_work_item_dims, NULL));
            // Get device max work item sizes in each dimension
            vector<size_t> work_item_sizes(max_work_item_dims);
            EXPECT_EQ(CL_SUCCESS, clGetDeviceInfo(cl_devices[j], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(work_item_sizes), work_item_sizes.data(), NULL));
            cout << "    Max workitem sizes:";
            for (size_t work_item_dim = 0; work_item_dim < max_work_item_dims; work_item_dim++)
            {
                cout << " " << work_item_sizes[work_item_dim];
            }
            cout << endl;

            cl_uint baseAddrAlign = 0;
            EXPECT_EQ(CL_SUCCESS, clGetDeviceInfo(cl_devices[j], CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(cl_uint), &baseAddrAlign, NULL));
            cout << "    Base address alignment: " << baseAddrAlign << " bits" << endl;

            // Get device image support
            cl_bool image_support;
            EXPECT_EQ(CL_SUCCESS, clGetDeviceInfo(cl_devices[j], CL_DEVICE_IMAGE_SUPPORT, sizeof(image_support), &image_support, NULL));
            cout << "    Image support: " << (image_support ? "Available" : "Not available") << endl;

            if (image_support)
            {

                size_t image_size;

                // Get device image 2d max width
                EXPECT_EQ(CL_SUCCESS, clGetDeviceInfo(cl_devices[j], CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(image_size), &image_size, NULL));
                cout << "      Max image2d width: " << image_size << endl;

                // Get device image 2d max height
                EXPECT_EQ(CL_SUCCESS, clGetDeviceInfo(cl_devices[j], CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(image_size), &image_size, NULL));
                cout << "      Max image2d height: " << image_size << endl;

                // Get device image 3d max width
                EXPECT_EQ(CL_SUCCESS, clGetDeviceInfo(cl_devices[j], CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof(image_size), &image_size, NULL));
                cout << "      Max image3d width: " << image_size << endl;

                // Get device image 3d max height
                EXPECT_EQ(CL_SUCCESS, clGetDeviceInfo(cl_devices[j], CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof(image_size), &image_size, NULL));
                cout << "      Max image3d height: " << image_size << endl;

                // Get device image 2d max depth
                EXPECT_EQ(CL_SUCCESS, clGetDeviceInfo(cl_devices[j], CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof(image_size), &image_size, NULL));
                cout << "      Max image3d depth: " << image_size << endl;

                
            }
        }
        cout << endl << endl;
    }
}

} // UT namespace
