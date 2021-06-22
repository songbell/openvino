// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <unordered_map>
#include <unordered_set>
#include <string>

#include <vpu/myriad_config.hpp>
#include <vpu/configuration.hpp>
#include <vpu/private_plugin_config.hpp>

#include <vpu/parsed_config_base.hpp>

#include <vpu/utils/perf_report.hpp>
#include <vpu/utils/logger.hpp>
#include <vpu/utils/enums.hpp>

namespace vpu {

class ParsedConfig : public ParsedConfigBase {
public:
    ParsedConfig() = default;
    ParsedConfig(const ParsedConfig&) = default;
    ParsedConfig& operator=(const ParsedConfig&) = default;
    ParsedConfig(ParsedConfig&&) = delete;
    ParsedConfig& operator=(ParsedConfig&&) = delete;

    const std::string& compilerLogFilePath() const {
        return _compilerLogFilePath;
    }

    const CompilationConfig& compileConfig() const {
        return _compileConfig;
    }

    CompilationConfig& compileConfig() {
        return _compileConfig;
    }

    bool printReceiveTensorTime() const {
        return _printReceiveTensorTime;
    }

    bool perfCount() const {
        return _perfCount;
    }

    PerfReport perfReport() const {
        return _perfReport;
    }

protected:
    const std::unordered_set<std::string>& getCompileOptions() const override;
    const std::unordered_set<std::string>& getRunTimeOptions() const override;
    const std::unordered_set<std::string>& getDeprecatedOptions() const override;
    void parse(const std::map<std::string, std::string>& config) override;

private:
    std::string _compilerLogFilePath;
    CompilationConfig _compileConfig;
    bool _printReceiveTensorTime = false;
    bool _perfCount              = false;
    PerfReport _perfReport = PerfReport::PerLayer;
};

}  // namespace vpu
