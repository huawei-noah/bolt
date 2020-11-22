// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _ALGORITHM_MAP_H
#define _ALGORITHM_MAP_H

#include <map>
#include <set>
#include "thread_affinity.h"
#include "op_type.h"
#include "types.h"

class AlgorithmMap {
public:
    AlgorithmMap(Arch arch, std::string modelName, std::string deviceName, DataType dt)
    {
        this->algorithmFileName = "algorithmInfo_";
        this->algorithmFileName += deviceName;
        this->algorithmFileName += "_";
        this->algorithmFileName += modelName;
        this->algorithmFileName += "_";
        this->algorithmFileName += std::to_string(arch);
        this->algorithmFileName += "_";
        this->algorithmFileName += std::to_string(dt);
        this->hasAlgorithmFile = false;
        this->arch = arch;
        this->commonAlgoFileName = "commonAlgoInfo_";
        this->commonAlgoFileName += deviceName;
        this->commonAlgoFileName += "_";
        this->commonAlgoFileName += std::to_string(arch);
        this->commonAlgoFileName += "_";
        this->commonAlgoFileName += std::to_string(dt);
        this->hasCommonAlgoFile = false;
    }

    void setAlgorithmInfoToMap(
        std::string name, I32 *algorithmArray, U32 arrayNum, bool commonAlgo = false)
    {
        std::string algoInfo = "/";
        for (U32 i = 0; i < arrayNum; i++) {
            algoInfo += std::to_string(algorithmArray[i]);
            algoInfo += "/";
        }
        if (!commonAlgo) {
            this->algorithmMap[name] = algoInfo;
        } else {
            this->commonAlgoMap[name] = algoInfo;
        }
    }

    bool getAlgorithmInfoFromMap(
        std::string name, I32 *algorithmArray, U32 arrayNum, bool commonAlgo = false)
    {
        std::string algoInfo;
        if (!commonAlgo) {
            if (this->algorithmMap.find(name) == this->algorithmMap.end()) {
                return false;
            }
            algoInfo = this->algorithmMap[name];
        } else {
            if (this->commonAlgoMap.find(name) == this->commonAlgoMap.end()) {
                return false;
            }
            algoInfo = this->commonAlgoMap[name];
        }
        U32 be = algoInfo.find_first_of("/");
        U32 end;
        for (U32 i = 0; i < arrayNum; i++) {
            end = algoInfo.find("/", be + 1);
            algorithmArray[i] = std::stoi(algoInfo.substr(be + 1, end - be - 1));
            be = end;
        }
        return true;
    }

    void loadAlgorithmMapFromFileStream(const char *algoFileStream)
    {
        U32 be = 0;
        be = readFileStreamForMap(algoFileStream, be, &this->algorithmMap);
#ifdef _USE_MALI
        be = readFileStreamForMap(algoFileStream, be, &this->kernelThreadMap);
#endif
        be = readFileStreamForMap(algoFileStream, be, &this->commonAlgoMap);
        if (algorithmMap.size()) {
            this->hasAlgorithmFile = true;
        }
        if (commonAlgoMap.size()) {
            this->hasCommonAlgoFile = true;
        }
    }

    void loadAlgorithmMapFromText(std::string algorithmMapPath)
    {
        if (algorithmMapPath == std::string("")) {
            UNI_DEBUG_LOG("load algorithm file failed, path is not set \n");
            return;
        }
        CI8 lastFlag = algorithmMapPath[algorithmMapPath.length() - 1];
        if (strcmp(&lastFlag, "/") != 0) {
            algorithmMapPath += "/";
        }
        this->hasAlgorithmFile = readTextForMap(algorithmFileName, algorithmMapPath, &algorithmMap);
        this->hasCommonAlgoFile =
            readTextForMap(commonAlgoFileName, algorithmMapPath, &commonAlgoMap);
    }

    void saveAlgorithmMapToText(std::string algorithmMapPath)
    {
        if (algorithmMapPath == std::string("")) {
            UNI_DEBUG_LOG("save algorithm file failed, path is not set \n");
            return;
        }
        if (this->hasAlgorithmFile) {
            return;
        }
        CI8 lastFlag = algorithmMapPath[algorithmMapPath.length() - 1];
        if (strcmp(&lastFlag, "/") != 0) {
            algorithmMapPath += "/";
        }
        saveMapToText(
            this->algorithmFileName, algorithmMapPath, this->algorithmMap, this->hasAlgorithmFile);
        saveMapToText(this->commonAlgoFileName, algorithmMapPath, this->commonAlgoMap,
            this->hasCommonAlgoFile);
    }

    void getCommonAlgoMapPara(U32 *ic_step,
        U32 *ihw_step,
        U32 *fn_step,
        U32 *ic_max,
        U32 *ihw_max,
        U32 *fn_max,
        std::set<U32> *fwh,
        std::set<U32> *stride)
    {
        if (ic_step) {
            *ic_step = 16;
        }
        if (ihw_step) {
            *ihw_step = 16;
        }
        if (fn_step) {
            *fn_step = 16;
        }
        if (ic_max) {
            *ic_max = 640;
        }
        if (ihw_max) {
            *ihw_max = 640;
        }
        if (fn_max) {
            *fn_max = 640;
        }
        if (fwh) {
            (*fwh).insert(1);
            (*fwh).insert(2);
            (*fwh).insert(3);
            (*fwh).insert(4);
            (*fwh).insert(5);
            (*fwh).insert(7);
        }
        if (stride) {
            (*stride).insert(1);
            (*stride).insert(2);
        }
    }

    void setCommonAlgoInfoToMap(OperatorType opType,
        DataType dt,
        U32 ic,
        U32 ih,
        U32 iw,
        U32 fn,
        U32 fh,
        U32 fw,
        U32 sh,
        U32 sw,
        I32 *algorithmArray,
        U32 arrayNum)
    {
        std::string algoName = getCommonAlgoName(opType, dt, ic, ih, iw, fn, fh, fw, sh, sw);
        setAlgorithmInfoToMap(algoName, algorithmArray, arrayNum, true);
    }

    bool getCommonAlgoInfoFromMap(OperatorType opType,
        DataType dt,
        U32 ic,
        U32 ih,
        U32 iw,
        U32 fn,
        U32 fh,
        U32 fw,
        U32 sh,
        U32 sw,
        I32 *algorithmArray,
        U32 arrayNum)
    {
        if (this->commonAlgoMap.size() == 0) {
            return false;
        }
        U32 ic_step, ihw_step, fn_step, ic_max, ihw_max, fn_max;
        std::set<U32> fwh;
        std::set<U32> stride;
        getCommonAlgoMapPara(
            &ic_step, &ihw_step, &fn_step, &ic_max, &ihw_max, &fn_max, &fwh, &stride);
        ic = ((ic + ic_step - 1) / ic_step) * ic_step;
        ih = ((ih + ihw_step - 1) / ihw_step) * ihw_step;
        iw = ((iw + ihw_step - 1) / ihw_step) * ihw_step;
        fn = ((fn + fn_step - 1) / fn_step) * fn_step;
        ic = (ic > ic_max) ? ic_max : ic;
        ih = (ih > ihw_max) ? ihw_max : ih;
        iw = (iw > ihw_max) ? ihw_max : iw;
        fn = (fn > fn_max) ? fn_max : fn;
        fw = (fw < fh) ? fh : fw;
        while (fwh.find(fw) == fwh.end()) {
            fw--;
        }
        while (stride.find(sw) == stride.end()) {
            sw--;
        }
        std::string algoName = getCommonAlgoName(opType, dt, ic, ih, iw, fn, fh, fw, sh, sw);
        return getAlgorithmInfoFromMap(algoName, algorithmArray, arrayNum, true);
    }

#ifdef _USE_MALI
    void setKernelThreadInfoToMap(std::string name, U32 gs[3], U32 ls[3])
    {
        std::string kernelThreadInfo = "/";
        for (U32 i = 0; i < 3; i++) {
            kernelThreadInfo += std::to_string(gs[i]);
            kernelThreadInfo += "/";
        }
        for (U32 i = 0; i < 3; i++) {
            kernelThreadInfo += std::to_string(ls[i]);
            kernelThreadInfo += "/";
        }
        kernelThreadMap[name] = kernelThreadInfo;
    }

    bool getKernelThreadInfoFromMap(std::string name, U32 *gs, U32 *ls)
    {
        bool findKernelInfo = kernelThreadMap.count(name);
        if (!findKernelInfo) {
            return findKernelInfo;
        }
        std::string kernelThreadInfo = kernelThreadMap[name];
        U32 be = kernelThreadInfo.find_first_of("/");
        U32 end;
        for (U32 i = 0; i < 3; i++) {
            end = kernelThreadInfo.find("/", be + 1);
            gs[i] = std::stoi(kernelThreadInfo.substr(be + 1, end - be - 1));
            be = end;
        }
        for (U32 i = 0; i < 3; i++) {
            end = kernelThreadInfo.find("/", be + 1);
            ls[i] = std::stoi(kernelThreadInfo.substr(be + 1, end - be - 1));
            be = end;
        }
        return findKernelInfo;
    }
#endif

private:
    U32 readFileStreamForMap(
        const char *algoFileStream, U32 be, std::map<std::string, std::string> *targetMap)
    {
        int num;
        std::string content(algoFileStream);
        std::string numString = "";
        std::string nameString = "";
        std::string infoString = "";
        while (content[be] == '\n' || content[be] == '\r' || content[be] == '\t' ||
            content[be] == ' ') {
            be++;
        }
        if (be >= content.size()) {
            return content.size();
        }
        if (content[be] == '\0') {
            return be;
        }
        while (content[be] != '\n') {
            numString += content[be];
            be++;
        }
        num = (numString.size()) ? std::stoi(numString) : 0;
        for (int i = 0; i < num; i++) {
            be++;
            while (content[be] != ' ') {
                nameString += content[be];
                be++;
            }
            be++;
            while (content[be] != '\n') {
                infoString += content[be];
                be++;
            }
            (*targetMap)[nameString] = infoString;
            nameString = "";
            infoString = "";
        }
        return be++;
    }

    bool readTextForMap(
        std::string fileName, std::string path, std::map<std::string, std::string> *targetMap)
    {
        std::string fullyFileName = path + fileName;
        FILE *file = fopen(fullyFileName.c_str(), "r");
        if (!file || feof(file)) {
            return false;
        }
        UNI_INFO_LOG("load algorithmFile %s\n", fullyFileName.c_str());
        int num = 0;
        fscanf(file, "%d", &num);
        char operatorName[100];
        char algorithm[100];
        for (int i = 0; i < num; i++) {
            fscanf(file, "%s %s", operatorName, algorithm);
            (*targetMap)[operatorName] = algorithm;
        }
#ifdef _USE_MALI
        if (this->arch == MALI && fileName == this->algorithmFileName) {
            fscanf(file, "%d", &num);
            char kernelName[100];
            char kernelThreadInfo[100];
            for (int i = 0; i < num; i++) {
                fscanf(file, "%s %s", kernelName, kernelThreadInfo);
                kernelThreadMap[kernelName] = kernelThreadInfo;
            }
        }
#endif
        fclose(file);
        return true;
    }

    void saveMapToText(std::string fileName,
        std::string path,
        std::map<std::string, std::string> targetMap,
        bool noNeedSave)
    {
        if (noNeedSave) {
            return;
        }
        if (targetMap.size() > 0) {
            std::string fullyFileName = path + fileName;
            UNI_DEBUG_LOG("save algorithmFile %s\n", fullyFileName.c_str());
            FILE *file = fopen(fullyFileName.c_str(), "w");
            fprintf(file, "%ld\n", (I64)targetMap.size());
            for (auto iter : targetMap) {
                fprintf(file, "%s %s\n", iter.first.c_str(), iter.second.c_str());
            }
#ifdef _USE_MALI
            if (this->arch == MALI && fileName == this->algorithmFileName) {
                fprintf(file, "%ld\n", (I64)kernelThreadMap.size());
                for (auto iter : kernelThreadMap) {
                    fprintf(file, "%s %s\n", iter.first.c_str(), iter.second.c_str());
                }
            }
#endif
            fclose(file);
        }
    }

    std::string getCommonAlgoName(
        OperatorType opType, DataType dt, U32 ic, U32 ih, U32 iw, U32 fn, U32 fh, U32 fw, U32 sh, U32 sw)
    {
        std::string algoName = "op" + std::to_string(opType) + "dt" + std::to_string(dt);
        algoName += "ic" + std::to_string(ic);
        algoName += "ih" + std::to_string(ih);
        algoName += "iw" + std::to_string(iw);
        algoName += "fn" + std::to_string(fn);
        algoName += "fh" + std::to_string(fh);
        algoName += "fw" + std::to_string(fw);
        algoName += "sh" + std::to_string(sh);
        algoName += "sw" + std::to_string(sw);
        return algoName;
    }

    std::map<std::string, std::string> algorithmMap;
    std::string algorithmFileName;
    Arch arch;
    bool hasAlgorithmFile;
#ifdef _USE_MALI
    std::map<std::string, std::string> kernelThreadMap;
#endif
    std::map<std::string, std::string> commonAlgoMap;
    std::string commonAlgoFileName;
    bool hasCommonAlgoFile;
};
#endif
