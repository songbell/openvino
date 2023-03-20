// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/detection_output.hpp"

#include "intel_gpu/primitives/detection_output.hpp"

namespace ov {
namespace intel_gpu {

static cldnn::prior_box_code_type PriorBoxCodeFromString(const std::string& str) {
    static const std::map<std::string, cldnn::prior_box_code_type> CodeNameToType = {
        { "caffe.PriorBoxParameter.CORNER" , cldnn::prior_box_code_type::corner },
        { "caffe.PriorBoxParameter.CENTER_SIZE" , cldnn::prior_box_code_type::center_size },
        { "caffe.PriorBoxParameter.CORNER_SIZE" , cldnn::prior_box_code_type::corner_size },
    };
    auto it = CodeNameToType.find(str);
    if (it != CodeNameToType.end()) {
        return it->second;
    } else {
        IE_THROW() << "Unknown Prior-Box code type: " << str;
    }
    return cldnn::prior_box_code_type::corner;
}

static void CreateDetectionOutputOp(Program& p, const std::shared_ptr<ngraph::op::v0::DetectionOutput>& op) {
    validate_inputs_count(op, {3});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    auto attrs = op->get_attrs();

    uint32_t num_classes            = attrs.num_classes;
    bool share_location             = attrs.share_location;
    int background_label_id         = attrs.background_label_id;
    float nms_threshold             = attrs.nms_threshold;
    int top_k                       = attrs.top_k;
    float confidence_threshold      = attrs.confidence_threshold;
    float eta                       = 1.0f;
    int keep_top_k                  = attrs.keep_top_k[0];
    bool variance_encoded_in_target = attrs.variance_encoded_in_target;
    int input_width                 = static_cast<int>(attrs.input_width);
    int input_height                = static_cast<int>(attrs.input_height);
    bool normalized                 = attrs.normalized;
    std::string code_type           = attrs.code_type;
    bool clip_before_nms            = attrs.clip_before_nms;
    bool clip_after_nms             = attrs.clip_after_nms;
    bool decrease_label_id          = attrs.decrease_label_id;

    cldnn::prior_box_code_type cldnnCodeType = PriorBoxCodeFromString(code_type);
    int32_t prior_info_size = normalized != 0 ? 4 : 5;
    int32_t prior_coordinates_offset = normalized != 0 ? 0 : 1;

    auto detectionPrim = cldnn::detection_output(layerName,
                                                 inputs[0],
                                                 inputs[1],
                                                 inputs[2],
                                                 num_classes,
                                                 keep_top_k,
                                                 share_location,
                                                 background_label_id,
                                                 nms_threshold,
                                                 top_k,
                                                 eta,
                                                 cldnnCodeType,
                                                 variance_encoded_in_target,
                                                 confidence_threshold,
                                                 prior_info_size,
                                                 prior_coordinates_offset,
                                                 normalized,
                                                 input_width,
                                                 input_height,
                                                 decrease_label_id,
                                                 clip_before_nms,
                                                 clip_after_nms);

    p.add_primitive(*op, detectionPrim);
}

REGISTER_FACTORY_IMPL(v0, DetectionOutput);

}  // namespace intel_gpu
}  // namespace ov
