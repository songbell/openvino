// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ie_icore.hpp"
#include "plugin.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_inference_plugin_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_icore.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iinference_plugin.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/util/common_util.hpp"
#include "common_test_utils/file_utils.hpp"
#include <iostream>

using ::testing::NiceMock;
using pluginConfigParams = std::tuple<
                        ov::AnyMap    // property to query the API
                        >;

class MockPluginBase : public ov::IPlugin {
public:
    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::string& model_path,
                                                      const ov::AnyMap& properties) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties,
                                                      const ov::RemoteContext& context) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    void set_property(const ov::AnyMap& properties) override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    std::shared_ptr<ov::IRemoteContext> create_context(const ov::AnyMap& remote_properties) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    std::shared_ptr<ov::IRemoteContext> get_default_context(const ov::AnyMap& remote_properties) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model, const ov::AnyMap& properties) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model,
                                                     const ov::RemoteContext& context,
                                                     const ov::AnyMap& properties) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                    const ov::AnyMap& properties) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) {
        return ov::Any{};
    }
};

class CoreIntegrationTest : public ::testing::TestWithParam<pluginConfigParams> {
public:
    std::shared_ptr<void> sharedObjectLoader;
    std::function<void(ov::IPlugin*)> injectProxyEngine;
    std::shared_ptr<NiceMock<MockPluginBase>> mockplugin;
    ov::AnyMap properties;
    ov::Core core;

private:
    template <class T>
    std::function<T> make_std_function(const std::string& functionName) {
        std::function<T> ptr(reinterpret_cast<T*>(ov::util::get_symbol(sharedObjectLoader, functionName.c_str())));
        return ptr;
    }

public:
    static std::string getTestCaseName(testing::TestParamInfo<pluginConfigParams> obj) {
        ov::AnyMap property;
        std::tie(property) = obj.param;
        std::ostringstream result;
        for (const auto& iter : property) {
            if (!iter.second.empty())
                result << "_" << iter.first << "_" << iter.second.as<std::string>() << "_";
            else
                result << "_" << iter.first << "_";
        }
        return result.str();
    }
    static std::string get_mock_engine_path() {
        std::string mockEngineName("mock_engine");
        return ov::util::make_plugin_library_name(CommonTestUtils::getExecutableDirectory(),
                                                  mockEngineName + IE_BUILD_POSTFIX);
    }
    void TearDown() override {
        mockplugin.reset();
        core.unload_plugin("MOCK_HARDWARE");
    }
    void SetUp() override {
        mockplugin = std::shared_ptr<NiceMock<MockPluginBase>>(new NiceMock<MockPluginBase>());
        std::string libraryPath = get_mock_engine_path();
        sharedObjectLoader = ov::util::load_shared_object(libraryPath.c_str());
        injectProxyEngine = make_std_function<void(ov::IPlugin*)>("CreatePluginEngine");
        std::tie(properties) = GetParam();
        mockplugin->set_device_name("MOCK_HARDWARE");
        injectProxyEngine(mockplugin.get());
        core.register_plugin(ov::util::make_plugin_library_name(CommonTestUtils::getExecutableDirectory(),
                                                             std::string("mock_engine") + IE_BUILD_POSTFIX),
                          "MOCK_HARDWARE");
        core.get_property("MOCK_HARDWARE", ov::num_streams);
    }
};