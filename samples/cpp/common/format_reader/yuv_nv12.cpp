// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#include <fstream>
#include <iostream>
#include <string>

#include "yuv_nv12.h"
// clang-format on

using namespace FormatReader;

YUV_NV12::YUV_NV12(const std::string& filename) {
    auto pos = filename.rfind('.');
    if (pos == std::string::npos)
        return;
    if (filename.substr(pos + 1) != "yuv")
        return;

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return;
    }

    file.seekg(0, file.end);
    _size = file.tellg();
    file.seekg(0, file.beg);

    _data.reset(new unsigned char[_size], std::default_delete<unsigned char[]>());

    file.read(reinterpret_cast<char*>(_data.get()), _size);

    file.close();
}
