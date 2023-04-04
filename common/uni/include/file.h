// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_FILE
#define _H_FILE

#include <vector>
#include <sys/stat.h>
#include <errno.h>
#if defined(__linux__) || defined(_WIN32)
#define _USE_DIRENT_H
#endif
#ifdef _USE_DIRENT_H
#include <dirent.h>
#endif
#include "string_functions.h"

#include "error.h"
#include "secure_c_wrapper.h"

inline int is_directory(const char *path)
{
    if (path == NULL) {
        return 0;
    }
    int ret = 0;
    struct stat s;
    if (stat(path, &s) == 0) {
        if (s.st_mode & S_IFDIR) {
            ret = 1;
        } else if (s.st_mode & S_IFREG) {
            ret = 0;
        } else {
            UNI_ERROR_LOG("can not recognize %s.\n", path);
        }
    }
    return ret;
}

inline int file_exists(const char *file)
{
    if (file == NULL) {
        return 0;
    }
    FILE *fp = fopen(file, "rb");
    if (fp == NULL) {
        return 0;
    }
    fclose(fp);
    return 1;
}

inline std::vector<std::string> search_files(
    std::string directory, std::string suffix = "", std::string have = "")
{
    std::vector<std::string> names;
    if (directory == "") {
        return names;
    }
#ifdef _USE_DIRENT_H
    DIR *handle = opendir(directory.c_str());
    struct dirent *dir;
    int suffix_len = suffix.size();
    int have_len = have.size();
    if (handle != NULL) {
        while ((dir = readdir(handle)) != NULL) {
            std::string name = dir->d_name;
            if (name == "." || name == "..") {
                continue;
            }
            if (is_directory((directory + "/" + name).c_str())) {
                continue;
            }
            int name_len = name.size();
            if (suffix_len > 0 && name_len > suffix_len) {
                if (name.substr(name_len - suffix_len) == suffix) {
                    names.push_back(name);
                }
            }
            if (have_len > 0 && name_len >= have_len) {
                if (name.find(have) != std::string::npos) {
                    names.push_back(name);
                }
            }
            if (suffix_len == 0 && have_len == 0) {
                names.push_back(name);
            }
        }
    } else {
        UNI_ERROR_LOG("can not access directory %s.\n", directory.c_str());
    }
    closedir(handle);
#else
    UNI_WARNING_LOG("deprecated api:%s.\n", __func__);
#endif
    return names;
}

inline EE load_binary(const char *file, void **content, size_t *size)
{
    if (file == NULL || content == NULL || size == NULL) {
        return NULL_POINTER;
    }
    *content = NULL;
    *size = 0;

    FILE *fp = fopen(file, "rb");
    if (fp == NULL) {
        int ret = errno;
        UNI_WARNING_LOG("can not load binary file %s, error:%d.\n", file, ret);
        return FILE_ERROR;
    }
    fseek(fp, 0, SEEK_END);
    *size = ftell(fp);
    rewind(fp);

    void *bytes = malloc(*size);
    if (bytes == NULL) {
        UNI_ERROR_LOG("allocate memory error.\n");
        return NULL_POINTER;
    }
    size_t length = fread(bytes, 1, *size, fp);
    if (length != *size) {
        UNI_ERROR_LOG("read binary file %s error: try to read %zu bytes, but actually %zu bytes.\n",
            file, *size, length);
        return FILE_ERROR;
    }
    fclose(fp);
    *content = bytes;
    return SUCCESS;
}

inline EE save_binary(const char *file, const void *content, size_t size)
{
    if (file == NULL) {
        return NULL_POINTER;
    }
    if (content == NULL) {
        return SUCCESS;
    }
    FILE *fp = fopen(file, "wb");
    if (!fp) {
        int ret = errno;
        UNI_ERROR_LOG("can not write binary to file %s, error:%d.\n", file, ret);
        return FILE_ERROR;
    }
    size_t length = fwrite(content, 1, size, fp);
    if (length != size) {
        UNI_ERROR_LOG("save binary file %s error: try to write %zu bytes, but actually %zu "
                      "bytes.\n",
            file, size, length);
        return FILE_ERROR;
    }
    fclose(fp);
    return SUCCESS;
}

inline EE load_floats(const char *file, float *content, size_t size)
{
    if (file == NULL) {
        return NULL_POINTER;
    }
    if (content == NULL) {
        return NULL_POINTER;
    }
    FILE *fp = fopen(file, "r");
    if (!fp) {
        int ret = errno;
        UNI_ERROR_LOG("can not read file %s, error:%d.\n", file, ret);
        return FILE_ERROR;
    }
    for (size_t i = 0; i < size; i++) {
        UNI_FSCANF(fp, "%f", content + i);
    }
    fclose(fp);
    return SUCCESS;
}

inline std::string load_string(const char *file)
{
    std::string ret = "";
    FILE *fp = fopen(file, "r");
    if (!fp) {
        UNI_ERROR_LOG("can not load string from file %s.\n", file);
    }
    char content[1024];
    while (fgets(content, 1024, fp) != NULL) {
        ret += content;
    }
    fclose(fp);
    return ret;
}

inline EE save_string(const char *file, const char *content)
{
    FILE *fp = fopen(file, "w");
    if (!fp) {
        int ret = errno;
        UNI_ERROR_LOG("can not write string to file %s, error:%d.\n", file, ret);
        return FILE_ERROR;
    }
    fprintf(fp, "%s", content);
    fclose(fp);
    return SUCCESS;
}

inline std::string hex_array(const unsigned char *data, size_t length)
{
    std::string ret = "{";
    char buffer[16];
    for (size_t i = 0; i < length; i++) {
        if (i % 20 == 0) {
            ret += "\n";
        }
        UNI_SNPRINTF(buffer, 16, "0x%02x", data[i]);
        ret += std::string(buffer);
        if (i != length - 1) {
            ret += ", ";
        }
    }
    ret += "}";
    return ret;
}
#endif
