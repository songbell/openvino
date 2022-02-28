// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_log_test.hpp"

INSTANTIATE_TEST_SUITE_P(
        accuracy, myriadLayersTestsLog_smoke,
        ::testing::ValuesIn(s_logParams));
