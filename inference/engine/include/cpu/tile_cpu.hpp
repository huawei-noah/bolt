// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _TILE_CPU_H
#define _TILE_CPU_H

#include "tile.hpp"

class TileCPU : public Tile {
public:
    TileCPU(DataType dt, TileParamSpec p) : Tile(dt, p)
    {}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<TileCPU> mem = std::shared_ptr<TileCPU>(new TileCPU(this->dt, this->p));
        *mem = *this;
        return mem;
    }

    TileParamSpec get_param(TensorDesc desc)
    {
        TileParamSpec ps = this->p;
        if (ps.num_repeats == 0) {
            ps.num_repeats = desc.dims[0];
            for (int i = 0; i < ps.num_repeats; i++) {
                ps.repeats[i] = desc.dims[desc.nDims + i];
            }
        }
        return ps;
    }

    void run() override
    {
        TileParamSpec ps = p;
        if (ps.num_repeats == 0 && this->inputTensors.size() > 1) {
            ps = get_param(this->inputTensors[1].get_desc());
        }
        CHECK_STATUS(tile(
            this->inputTensors[0], ps, this->temp, this->outputTensors[0], &this->archInfo));
        this->outputTensors[0].set_scale(this->inputTensors[0].get_scale());
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        TileParamSpec ps = p;
        if (ps.num_repeats == 0 && inTensors.size() > 1) {
            ps = get_param(inTensors[1]->get_desc());
        }
        return tile_infer_output_size(inTensors[0], ps, outTensors[0], &this->archInfo);
    }
};

#endif  // _TILECPU_H
