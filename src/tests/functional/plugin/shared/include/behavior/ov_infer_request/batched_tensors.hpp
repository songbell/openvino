// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>
#include <string>
#include "functional_test_utils/ov_plugin_cache.hpp"
#include <base/behavior_test_utils.hpp>

namespace ov {
namespace test {
namespace behavior {

class OVInferRequestBatchedTests : public testing::WithParamInterface<std::string>,
                                   public CommonTestUtils::TestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<std::string>& device_name);

protected:
    void SetUp() override;

    void TearDown() override;

    static std::string generateCacheDirName(const std::string& test_name);
    static std::shared_ptr<Model> create_n_inputs(size_t num, element::Type type,
                                                  const PartialShape& shape, const ov::Layout& layout);

    std::shared_ptr<runtime::Core> ie = utils::PluginCache::get().core();
    std::string targetDevice;
    std::string m_cache_dir; // internal member
    bool m_need_reset_core = false;
};

}  // namespace behavior
}  // namespace test
}  // namespace ov
