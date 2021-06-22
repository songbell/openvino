// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/parsed_config_base.hpp>

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <string>
#include <memory>
#include <map>

#include <ie_plugin_config.hpp>

namespace vpu {

const std::unordered_map<std::string, bool> ParsedConfigBase::switches = {
    { CONFIG_VALUE(YES), true },
    { CONFIG_VALUE(NO), false }
};

ParsedConfigBase::ParsedConfigBase() {
    _log = std::make_shared<Logger>("Config", LogLevel::Warning, consoleOutput());
}

ParsedConfigBase::~ParsedConfigBase() = default;

void ParsedConfigBase::update(
        const std::map<std::string, std::string>& config,
        ConfigMode mode) {
    const auto& compileOptions = getCompileOptions();
    const auto& runTimeOptions = getRunTimeOptions();
    const auto& deprecatedOptions = getDeprecatedOptions();

    for (const auto& entry : config) {
        const bool isCompileOption = compileOptions.count(entry.first) != 0;
        const bool isRunTimeOption = runTimeOptions.count(entry.first) != 0;
        const bool isDeprecatedOption = deprecatedOptions.count(entry.first) != 0;

        if (!isCompileOption && !isRunTimeOption) {
            IE_THROW(NotFound) << entry.first
                    << " key is not supported for VPU";
        }

        if (mode == ConfigMode::RunTime) {
            if (!isRunTimeOption) {
                _log->warning("%s option is used in %s mode", entry.first, mode);
            }
        }

        if (isDeprecatedOption) {
            _log->warning("Deprecated option was used : %s", entry.first);
        }
    }

    parse(config);
}

const std::unordered_set<std::string>& ParsedConfigBase::getCompileOptions() const {
    static const std::unordered_set<std::string> options;
    return options;
}

const std::unordered_set<std::string>& ParsedConfigBase::getRunTimeOptions() const {
IE_SUPPRESS_DEPRECATED_START
    static const std::unordered_set<std::string> options = {
        CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS),
    };
IE_SUPPRESS_DEPRECATED_END

    return options;
}

const std::unordered_set<std::string>& ParsedConfigBase::getDeprecatedOptions() const {
    static const std::unordered_set<std::string> options;
    return options;
}

void ParsedConfigBase::parse(const std::map<std::string, std::string>& config) {
    setOption(_exclusiveAsyncRequests, switches, config, CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS));
}

std::unordered_set<std::string> ParsedConfigBase::merge(
            const std::unordered_set<std::string>& set1,
            const std::unordered_set<std::string>& set2) {
    auto out = set1;
    out.insert(set2.begin(), set2.end());
    return out;
}

void ParsedConfigBase::setOption(std::string& dst, const std::map<std::string, std::string>& config, const std::string& key) {
    const auto value = config.find(key);
    if (value != config.end()) {
        dst = value->second;
    }
}

}  // namespace vpu
