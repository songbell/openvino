/* ============================================================================
 * INTEL CONFIDENTIAL
 *
 * Copyright 2021 Intel Corporation All Rights Reserved.
 *
 * The source code contained or described herein and all documents related to
 * the source code ("Material") are owned by Intel Corporation or its suppliers
 * or licensors. Title to the Material remains with Intel Corporation or its
 * suppliers and licensors. The Material contains trade secrets and proprietary
 * and confidential information of Intel or its suppliers and licensors. The
 * Material is protected by worldwide copyright and trade secret laws and
 * treaty provisions. No part of the Material may be used, copied, reproduced,
 * modified, published, uploaded, posted, transmitted, distributed, or
 * disclosed in any way without Intel's prior express written permission.
 *
 * No license under any patent, copyright, trade secret or other intellectual
 * property right is granted to or conferred upon you by disclosure or delivery
 * of the Materials, either expressly, by implication, inducement, estoppel or
 * otherwise. Any license under such intellectual property rights must be
 * express and approved by Intel in writing.
 * ============================================================================
 * Shared under CNDA#582531
 * ============================================================================
 */

#include <inference_engine.hpp>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <fstream>
#include <unordered_map>

#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset2.hpp"
#include "ngraph/opsets/opset3.hpp"

#include "transformations/serialize.hpp"
#include "transformations/common_optimizations/common_optimizations.hpp"
#include "conv2d_decomposition.hpp"

using namespace InferenceEngine;
using namespace ngraph;
using namespace op;

typedef enum {
    AF_NONE = 0,
    AF_RELU,
    AF_PRELU,
    AF_SIGMOID
} AF;

std::vector<float> ReorderNCWH2NCHW(std::vector<float>& weights_ncwh, Shape filter_shape_nchw)
{
    if (shape_size(filter_shape_nchw) != weights_ncwh.size())
    {
        printf("ERROR: size of weights does not match shape");
    }
    std::vector<float> weights_nchw(weights_ncwh.size());
    const size_t CHW = filter_shape_nchw[1] * filter_shape_nchw[2] * filter_shape_nchw[3];
    const size_t HW = filter_shape_nchw[2] * filter_shape_nchw[3];
    const size_t W = filter_shape_nchw[3];
    const size_t H = filter_shape_nchw[2];

    for (size_t n = 0; n < filter_shape_nchw[0]; n++) {
        for (size_t c = 0; c < filter_shape_nchw[1]; c++) {
            size_t nc_index = n * CHW + c * HW;
            for (size_t w = 0; w < W; w++) {
                for (size_t h = 0; h < H; h++) {
                    weights_nchw[nc_index + h * W + w] = weights_ncwh[nc_index + w * H + h];
                }
            }
        }
    }
    return weights_nchw;
}

class ParameterProvider
{
public:
    virtual std::vector<float>GetParametersAsFlatBufferByName(std::string name) = 0;
    virtual ~ParameterProvider() {}
};

typedef std::vector<float>(*weight_provider_f)(std::string name);

std::shared_ptr<Node>CreateStreamingBufferConcat(const Output<Node>& input, ngraph::Shape memory_shape, int axis, std::string memory_buffer_name, SinkVector& sinks)
{
    if (memory_shape.size() != 4 || axis != 1)
        return nullptr;

    size_t flat_memory_size = shape_size(memory_shape);
    std::vector<float> initial_memory_state_data(flat_memory_size);

    std::shared_ptr<Node> initial_memory_state_const = op::Constant::create(
        element::Type_t::f32, memory_shape, initial_memory_state_data);

    auto read_value_node = std::make_shared<ngraph::opset3::ReadValue>(
        initial_memory_state_const,
        memory_buffer_name);

    OutputVector concat_inputs = { read_value_node, input };
    auto concat = std::make_shared<opset1::Concat>(concat_inputs, axis);

    auto input_shape = input.get_shape();

    size_t offset = input_shape[1];
    size_t size = memory_shape[1];
    auto state_to_be_stored = std::make_shared<ngraph::opset1::StridedSlice>(
        concat->output(0), // data
        op::Constant::create(ngraph::element::i64, ngraph::Shape{ memory_shape.size() }, { 0ull, offset, 0ull, 0ull }), // begin sice index
        op::Constant::create(ngraph::element::i64, ngraph::Shape{ memory_shape.size() }, { 0ull, offset + size, input_shape[2], input_shape[3] }), // end slice index
        op::Constant::create(ngraph::element::i64, ngraph::Shape{ memory_shape.size() }, { 1ull , 1ull , 1ull , 1ull }), // strides
        std::vector<int64_t>{1, 0, 0, 0}, // begin mask
        std::vector<int64_t>{1, 0, 0, 0} // end mask
    );

    auto store_value_node = std::make_shared<ngraph::opset3::Assign>(
        state_to_be_stored->output(0),
        memory_buffer_name);
    sinks.push_back(store_value_node);
    return concat;
}

std::shared_ptr<Node>CreateWrappedConvolutionAddSigmoidVariadicSplitMul(
    const Output<Node>& input,
    std::vector<ptrdiff_t>pads_begin,
    std::vector<ptrdiff_t>pads_end,
    SizeVector dilations,
    Shape filter_size,
    std::string conv_name,
    ParameterProvider* provider)
{
    auto weights = provider->GetParametersAsFlatBufferByName(conv_name);
    weights = ReorderNCWH2NCHW(weights, filter_size);
    auto bias = provider->GetParametersAsFlatBufferByName(conv_name+"/bias");

    ngraph::Shape half_filter_size = { filter_size[0] / 2, filter_size[1], filter_size[2], filter_size[3] };

    std::vector<float> first_half_weights(weights.begin(), weights.begin() + weights.size() / 2);
    std::vector<float> second_half_weights(weights.begin() + weights.size() / 2, weights.end());

    std::shared_ptr<Node> conv_filters_first_half_const = op::Constant::create(
        element::Type_t::f32, half_filter_size, first_half_weights);

    std::shared_ptr<Node> conv_filters_second_half_const = op::Constant::create(
        element::Type_t::f32, half_filter_size, second_half_weights);

    std::shared_ptr<Node> leading_permute_1 = std::make_shared<op::Transpose>(input,
        op::Constant::create(element::Type_t::i64, Shape{ 4 }, { 0ll, 3ll, 1ll, 2ll })->output(0));

    std::shared_ptr<Node> leading_permute_2 = std::make_shared<op::Transpose>(input,
        op::Constant::create(element::Type_t::i64, Shape{ 4 }, { 0ll, 3ll, 1ll, 2ll })->output(0));

    std::shared_ptr<Node> first_conv = std::make_shared<op::v1::Convolution>(
        leading_permute_1->output(0),
        conv_filters_first_half_const->output(0),
        Strides(SizeVector{ 1, 1 }),
        CoordinateDiff(pads_begin), CoordinateDiff(pads_end),
        Strides(dilations));

    first_conv->set_friendly_name(conv_name+"/1");

    std::shared_ptr<Node> second_conv = std::make_shared<op::v1::Convolution>(
        leading_permute_2->output(0),
        conv_filters_second_half_const->output(0),
        Strides(SizeVector{ 1, 1 }),
        CoordinateDiff(pads_begin), CoordinateDiff(pads_end),
        Strides(dilations));

    second_conv->set_friendly_name(conv_name + "/2");

    std::shared_ptr<Node> sigmoid;
    std::shared_ptr<Node> mul_node;
    if (bias.size()) {
        std::shared_ptr<Node> first_conv_bias_const = op::Constant::create(
            element::Type_t::f32, { 1,half_filter_size[0],1,1 }, std::vector<float>(bias.begin(), bias.begin()+ bias.size()/2));

        std::shared_ptr<Node> second_conv_bias_const = op::Constant::create(
            element::Type_t::f32, { 1,half_filter_size[0],1,1 }, std::vector<float>(bias.begin() + bias.size() / 2, bias.end()));

        std::shared_ptr<Node> first_add_bias = std::make_shared<op::v1::Add>(
            first_conv->output(0),
            first_conv_bias_const->output(0));
        first_add_bias->set_friendly_name(conv_name + "_1/bias");

        std::shared_ptr<Node> second_add_bias = std::make_shared<op::v1::Add>(
            second_conv->output(0),
            second_conv_bias_const->output(0));
        second_add_bias->set_friendly_name(conv_name + "_2/bias");

        sigmoid = std::make_shared<op::v0::Sigmoid>(second_add_bias->output(0));
        sigmoid->set_friendly_name(conv_name + "/sigmoid");

        std::shared_ptr<Node> trailing_permute_1 = std::make_shared<op::Transpose>(first_add_bias->output(0),
            op::Constant::create(element::Type_t::i64, Shape{ 4 }, { 0ll, 2ll, 3ll, 1ll })->output(0));
        std::shared_ptr<Node> trailing_permute_2 = std::make_shared<op::Transpose>(sigmoid->output(0),
            op::Constant::create(element::Type_t::i64, Shape{ 4 }, { 0ll, 2ll, 3ll, 1ll })->output(0));
        mul_node = std::make_shared <ngraph::opset1::Multiply>(trailing_permute_1->output(0), trailing_permute_2->output(0));
    } else {
        sigmoid = std::make_shared<op::v0::Sigmoid>(second_conv->output(0));
        sigmoid->set_friendly_name(conv_name + "/sigmoid");

        std::shared_ptr<Node> trailing_permute_1 = std::make_shared<op::Transpose>(first_conv->output(0),
            op::Constant::create(element::Type_t::i64, Shape{ 4 }, { 0ll, 2ll, 3ll, 1ll })->output(0));
        std::shared_ptr<Node> trailing_permute_2 = std::make_shared<op::Transpose>(sigmoid->output(0),
            op::Constant::create(element::Type_t::i64, Shape{ 4 }, { 0ll, 2ll, 3ll, 1ll })->output(0));
        mul_node = std::make_shared <ngraph::opset1::Multiply>(trailing_permute_1->output(0), trailing_permute_2->output(0));
    }

    return mul_node;
}

std::shared_ptr<Node> Downsample2xW_NHWC_C16(
    const Output<Node>& input,
    std::string conv_name,
    ParameterProvider* provider)
{
    auto weights = provider->GetParametersAsFlatBufferByName(conv_name);

    // first we need to align to 64 bytes
    // we have 16 channels... we need 32
    auto input_shape = input.get_shape();

    Output<Node> padded_input(input);
    auto padding_shape = { input_shape[0], input_shape[1], (4 - input_shape[2] % 4), input_shape[3] };
    // we need to concat input with zero buffer
    if (input_shape[2] % 4) {
        padded_input = std::make_shared<op::Concat>(OutputVector{
            input,
            op::Constant::create(ngraph::element::f32, padding_shape, std::vector<float>(shape_size(padding_shape)))->output(0) },
            2)->output(0);
    }
    auto padded_input_shape = padded_input.get_shape();
    size_t padded_input_size = shape_size(padded_input_shape);

    auto leading_reshape = std::make_shared<ngraph::opset1::Reshape>(
        padded_input,
        op::Constant::create(ngraph::element::i64, Shape{ 2 }, { 2ull, padded_input_size /2})->output(0),
        false);

    auto leading_transpose = std::make_shared<ngraph::opset1::Transpose>(
        leading_reshape->output(0),
        op::Constant::create(ngraph::element::i64, Shape{ 2 }, { 1, 0 })->output(0));

    // OK we have 32 channels
    auto reshape_to_2x_channels = std::make_shared<ngraph::opset1::Reshape>(
        leading_transpose->output(0),
        op::Constant::create(ngraph::element::i64, Shape{ 4 }, { padded_input_shape[0], padded_input_shape[1], padded_input_shape[2] / 2, padded_input_shape[3] * 2 })->output(0),
        false);

    // create new weights
    std::vector<float> w0_weights(weights.size());
    std::vector<float> w1_weights(weights.size());
    for (size_t i = 0; i < weights.size() / 2; i++) {
        w0_weights[2 * i] = weights[2*i];
        w0_weights[2 * i+1] = weights[2*i];
        w1_weights[2 * i] = weights[2*i+1];
        w1_weights[2 * i+1] = weights[2*i+1];
    }

    auto kernel_w0_const = op::Constant::create(ngraph::element::f32, Shape{ 1, 1, 1, padded_input_shape[3] * 2 }, w0_weights);
    auto kernel_w1_const = op::Constant::create(ngraph::element::f32, Shape{ 1, 1, 1, padded_input_shape[3] * 2 }, w1_weights);

    std::vector<int64_t> splits;
    for (unsigned int i = 0; i < padded_input_shape[2] / 2; i++)
        splits.push_back(1);

    auto variadic_split = std::make_shared<ngraph::opset1::VariadicSplit>(
        reshape_to_2x_channels->output(0),
        op::Constant::create(ngraph::element::i64, Shape{ 1 }, { 2 })->output(0),
        op::Constant::create(ngraph::element::i64, Shape{ splits.size() }, splits)->output(0));

    OutputVector chunks;
    // now we will implement grouped convolution as elementwise mul/add
    for (size_t w = 0; w < variadic_split->outputs().size(); w+=2)
    {
        auto mul_a0xw0 = std::make_shared <opset1::Multiply>(variadic_split->output(w), kernel_w0_const->output(0));
        mul_a0xw0->set_friendly_name(conv_name + "/mul_K0_W" + std::to_string(w));
        auto mul_a1xw1 = std::make_shared <opset1::Multiply>(variadic_split->output(w+1), kernel_w1_const->output(0));
        mul_a1xw1->set_friendly_name(conv_name + "/mul_K1_W" + std::to_string(w+1));
        auto add = std::make_shared <opset1::Add>(mul_a0xw0->output(0), mul_a1xw1->output(0));
        mul_a1xw1->set_friendly_name(conv_name + "/add" + std::to_string(w + 1));
        chunks.push_back(add);
    }

    auto concat = std::make_shared<op::Concat>(chunks, 3);

    auto back_to_orginal_ch_count_reshape = std::make_shared<ngraph::opset1::Reshape>(
        concat->output(0),
        op::Constant::create(ngraph::element::i64, Shape{ 2 }, { padded_input_size/4, 2ull })->output(0),
        false);

    auto trailing_transpose = std::make_shared<ngraph::opset1::Transpose>(
        back_to_orginal_ch_count_reshape,
        op::Constant::create(ngraph::element::i64, Shape{ 2 }, { 1, 0 })->output(0));

    auto trailing_reshape = std::make_shared<ngraph::opset1::Reshape>(
        trailing_transpose->output(0),
        op::Constant::create(ngraph::element::i64, Shape{ 4 }, { padded_input_shape[0], padded_input_shape[1], padded_input_shape[2]/2, padded_input_shape[3] })->output(0),
        false);
    if (shape_size(input_shape) == shape_size(padded_input_shape)) {
        return trailing_reshape;
    } else {
        std::vector<size_t> last_split{input_shape[2]/2, (padded_input_shape[2] - input_shape[2])/2 };
        auto truncated_padding = std::make_shared<ngraph::opset1::VariadicSplit>(
            trailing_reshape->output(0),
            op::Constant::create(ngraph::element::i64, Shape{ 1 }, { 2 })->output(0),
            op::Constant::create(ngraph::element::i64, Shape{ last_split.size() }, last_split)->output(0));
        return truncated_padding;
    }
}

std::shared_ptr<Node> Upsample2xW_NHWC(
    const Output<Node>& input,
    std::string conv_name,
    ParameterProvider* provider)
{
    auto weights = provider->GetParametersAsFlatBufferByName(conv_name);

    auto input_shape = input.get_shape();
    auto conv_filters_const = op::Constant::create(ngraph::element::f32, Shape{ 2*input_shape[3],input_shape[3],1,1}, weights);

    auto leading_transpose = std::make_shared<ngraph::opset1::Transpose>(
        input,
        op::Constant::create(ngraph::element::i64, Shape{ 4 }, { 0, 3, 1, 2 })->output(0));

    std::shared_ptr<Node> conv = std::make_shared<op::v1::Convolution>(
        leading_transpose->output(0),
        conv_filters_const->output(0),
        Strides(SizeVector{ 1, 1 }),
        CoordinateDiff({ 0,0 }), CoordinateDiff({ 0,0 }),
        Strides({ 1,1 }));

    conv->set_friendly_name(conv_name);

    auto trailing_transpose = std::make_shared<ngraph::opset1::Transpose>(
        conv->output(0),
        op::Constant::create(ngraph::element::i64, Shape{ 4 }, { 0, 2, 3, 1 })->output(0));

    auto trailing_reshape = std::make_shared<ngraph::opset1::Reshape>(
        trailing_transpose->output(0),
        op::Constant::create(ngraph::element::i64, Shape{ 4 }, { input_shape[0], input_shape[1], input_shape[2]*2, input_shape[3] })->output(0),
        false);
    return trailing_reshape;
}

std::vector<float> SpitWeightsInputChannelWise_NCHW(Shape NCHW_filter_size, std::vector<float> weights, size_t channel_first, size_t channel_count)
{
    std::vector<float> slitted_weights(channel_count * NCHW_filter_size[0]* NCHW_filter_size[2]* NCHW_filter_size[3]);
    size_t HW_size = NCHW_filter_size[2] * NCHW_filter_size[3];
    int index = 0;
    for (size_t k = 0; k < NCHW_filter_size[0]; k++) {
        for (size_t c = channel_first; c < channel_first + channel_count; c++) {
            size_t plane_index = (k * NCHW_filter_size[1] + c) * HW_size;
            for (size_t hw = 0; hw < HW_size; hw++) {
                slitted_weights[index++] = weights[plane_index + hw];
            }
        }
    }
    return slitted_weights;
}

std::vector<float> SpitWeightsOutputChannelWise_NCHW(Shape NCHW_filter_size, std::vector<float> weights, size_t out_channel_first, size_t out_channel_count)
{
    std::vector<float> slitted_weights(out_channel_count * NCHW_filter_size[1] * NCHW_filter_size[2] * NCHW_filter_size[3]);
    size_t HW_size = NCHW_filter_size[2] * NCHW_filter_size[3];
    int index = 0;
    for (size_t n = out_channel_first; n < out_channel_first + out_channel_count; n++) {
        for (size_t c = 0; c < NCHW_filter_size[1]; c++) {
            size_t plane_index = n * NCHW_filter_size[1] * HW_size;
            for (size_t hw = 0; hw < HW_size; hw++) {
                slitted_weights[index++] = weights[plane_index + hw];
            }
        }
    }
    return slitted_weights;
}

std::shared_ptr<Node>CreateConvolutionAfterChannelWiseConcatAddBiasNActivationFunction(
    const OutputVector& inputs,
    std::vector<ptrdiff_t>pads_begin,
    std::vector<ptrdiff_t>pads_end,
    SizeVector dilations,
    Shape filter_size,
    std::string conv_name,
    AF af,
    ParameterProvider* provider)
{
    Shape org_filter_size = filter_size;
    // minimum number of output channels is 4
    // as currently the transpose is supported for 8 columns - we use 8 output channels
    auto weights = provider->GetParametersAsFlatBufferByName(conv_name);
    if (org_filter_size[0] % 8 && org_filter_size[0] < 8) {
        size_t pad = 8 - org_filter_size[0] % 8;
        filter_size[0] = org_filter_size[0] + pad;
        std::vector<float> padding(pad * org_filter_size[1]* org_filter_size[2]* org_filter_size[3]);
        weights.insert(weights.end(), padding.begin(), padding.end());
    }
    weights = ReorderNCWH2NCHW(weights, filter_size);
    auto bias = provider->GetParametersAsFlatBufferByName(conv_name+"/bias");
    auto prelu_slope = af == AF_PRELU ? provider->GetParametersAsFlatBufferByName(conv_name + "/prelu") : std::vector<float>();

    std::vector<std::shared_ptr<Node>> concat_paths;
    size_t offset = 0;
    for (size_t i = 0; i < inputs.size(); i++)
    {
        auto const_NCHW_to_NHWC = op::Constant::create(element::Type_t::i64, Shape{ 4 }, { 0ll, 2ll, 3ll, 1ll });
        auto const_NHWC_to_NCHW = op::Constant::create(element::Type_t::i64, Shape{ 4 }, { 0ll, 3ll, 1ll, 2ll });

        // inputs are in NHWC
        auto input_channel_count = inputs[i].get_shape()[3];
        auto splitted_weights = SpitWeightsInputChannelWise_NCHW(filter_size, weights, offset, input_channel_count);
        std::shared_ptr<Node> conv_filters_const = op::Constant::create(
            element::Type_t::f32, { filter_size[0],input_channel_count, filter_size[2], filter_size[3] }, splitted_weights);
        offset += input_channel_count;

        std::shared_ptr<Node> leading_permute = std::make_shared<op::Transpose>(inputs[i],
            const_NHWC_to_NCHW->output(0));

        std::shared_ptr<Node> conv = std::make_shared<op::v1::Convolution>(
            leading_permute->output(0),
            conv_filters_const->output(0),
            Strides(SizeVector{ 1, 1 }),
            CoordinateDiff(pads_begin), CoordinateDiff(pads_end),
            Strides(dilations));

        conv->set_friendly_name(conv_name+"/"+std::to_string(i));

        std::shared_ptr<Node> trailing_permute;
        if (i == 0 && bias.size()) {
            std::shared_ptr<Node> conv_bias_const = op::Constant::create(
                element::Type_t::f32, { 1,filter_size[0],1,1 }, bias);
            std::shared_ptr<Node> add_bias = std::make_shared<op::v1::Add>(
                conv->output(0),
                conv_bias_const->output(0));
            add_bias->set_friendly_name(conv_name + "/" + std::to_string(i) + "/bias");
            if (af != AF_NONE && inputs.size() == 1) {
                std::shared_ptr<Node> af_inst;
                if (af == AF_RELU) {
                    af_inst = std::make_shared<op::Relu>(add_bias->output(0));
                    af_inst->set_friendly_name(conv_name + "/relu");
                } else if (af == AF_PRELU) {
                    auto prelu_slope_const = op::Constant::create(
                        element::Type_t::f32, { 1 }, prelu_slope);
                    af_inst = std::make_shared<op::v0::PRelu>(add_bias->output(0), prelu_slope_const);
                    af_inst->set_friendly_name(conv_name + "/prelu");
                } else if (af == AF_SIGMOID) {
                    af_inst = std::make_shared<op::v0::Sigmoid>(add_bias->output(0));
                    af_inst->set_friendly_name(conv_name + "/sigmoid");
                }

                trailing_permute = std::make_shared<op::Transpose>(af_inst->output(0),
                    const_NCHW_to_NHWC->output(0));
            } else {
                trailing_permute = std::make_shared<op::Transpose>(add_bias->output(0),
                    const_NCHW_to_NHWC->output(0));
            }
        } else {
            if (af != AF_NONE && inputs.size() == 1) {
                std::shared_ptr<Node> af_inst;
                if (af == AF_RELU) {
                    af_inst = std::make_shared<op::Relu>(conv->output(0));
                    af_inst->set_friendly_name(conv_name + "/relu");
                } else if (af == AF_PRELU) {
                    auto prelu_slope_const = op::Constant::create(
                        element::Type_t::f32, { 1 }, prelu_slope);
                    af_inst = std::make_shared<op::v0::PRelu>(conv->output(0), prelu_slope_const);
                    af_inst->set_friendly_name(conv_name + "/prelu");
                } else if (af == AF_SIGMOID) {
                    af_inst = std::make_shared<op::v0::Sigmoid>(conv->output(0));
                    af_inst->set_friendly_name(conv_name + "/sigmoid");
                }
                trailing_permute = std::make_shared<op::Transpose>(af_inst->output(0),
                    const_NCHW_to_NHWC->output(0));
            } else {
                trailing_permute = std::make_shared<op::Transpose>(conv->output(0),
                    const_NCHW_to_NHWC->output(0));
            }
        }

        concat_paths.push_back(trailing_permute);
    }

    std::shared_ptr<Node> result;

    std::shared_ptr<Node> conv_result = concat_paths[0];
    for (size_t i = 1; i < concat_paths.size(); i++) {
        auto add_result = std::make_shared<ngraph::opset1::Add>(concat_paths[i]->output(0), conv_result);
        conv_result = add_result;
    }
    if (inputs.size() > 1 && af != AF_NONE) {
        std::shared_ptr<Node> af_inst;
        if (af == AF_RELU) {
            af_inst = std::make_shared<op::Relu>(conv_result->output(0));
            af_inst->set_friendly_name(conv_name + "/relu");
        }
        else if (af == AF_PRELU) {
            auto prelu_slope_const = op::Constant::create(
                element::Type_t::f32, { 1 }, prelu_slope);
            af_inst = std::make_shared<op::v0::PRelu>(conv_result->output(0), prelu_slope_const);
            af_inst->set_friendly_name(conv_name + "/prelu");
        }
        else if (af == AF_SIGMOID) {
            af_inst = std::make_shared<op::v0::Sigmoid>(conv_result->output(0));
            af_inst->set_friendly_name(conv_name + "/sigmoid");
        }
        result = af_inst;
    } else {
        result = conv_result;
    }
    if (org_filter_size[0] % 8 && org_filter_size[0] < 8) {
        auto const_NCHW_to_NHWC = op::Constant::create(element::Type_t::i64, Shape{ 4 }, { 0ll, 2ll, 3ll, 1ll });
        auto const_NHWC_to_NCHW = op::Constant::create(element::Type_t::i64, Shape{ 4 }, { 0ll, 3ll, 1ll, 2ll });

        //size_t pad = 8 - org_filter_size[0] % 8;
        auto deinterleave_permute = std::make_shared<op::Transpose>(result->output(0),
            const_NHWC_to_NCHW->output(0));

        const auto axis_node = ngraph::opset1::Constant::create(element::i64, Shape{}, { 1 });
        const auto split = std::make_shared<ngraph::opset1::Split>(deinterleave_permute->output(0), axis_node, 8);
        if (org_filter_size[0] == 1) {
            auto conv_output_shape = conv_result->get_shape();
            auto reshape_to_NHWC = std::make_shared<opset1::Reshape>(split->output(0),
                op::Constant::create(ngraph::element::i64, Shape{ 4 }, { conv_output_shape[0], conv_output_shape[1], conv_output_shape[2], 1ull}), false);
            result = reshape_to_NHWC;
        } else {
            OutputVector ov;
            for (size_t i = 0; i < org_filter_size[0]; i++)
                ov.push_back(split->output(i));
            auto concat = std::make_shared<op::Concat>(ov, 1ll);
            result = std::make_shared<op::Transpose>(result->output(0),
                const_NCHW_to_NHWC->output(0));
        }
    }

    return result;
}

std::shared_ptr<Node> CreatePointwiseConvNHWC2NHCWUsingMatMul(
    const Output<Node>& input,
    std::string conv_name,
    ParameterProvider* provider)
{
    auto weights = provider->GetParametersAsFlatBufferByName(conv_name);

    auto input_shape = input.get_shape();
    if (input_shape.size() != 4)
        return nullptr;
    std::shared_ptr<op::Constant> matmul_weights_const;

    Output<Node> padded_input = input;
    size_t pad_size = 8 - (input_shape[2] & 7);
    size_t org_width = input_shape[2];
    size_t new_width = input_shape[2];
    if (input_shape[2] & 7) {
        auto pad_const = op::Constant::create(ngraph::element::f32,
            Shape{ input_shape[0],input_shape[1], pad_size, input_shape[3] },
            std::vector<float>(pad_size * input_shape[3]));
        padded_input = std::make_shared<opset1::Concat>(OutputVector{ input, pad_const->output(0) }, 2);
        new_width += pad_size;

        // pad weights
        auto new_weights = std::vector<float>(new_width * new_width);
        for (size_t r = 0; r < new_width; r++) {
            for (size_t c = 0; c < new_width; c++) {
                new_weights[r * org_width + c] = (c < org_width&& r < org_width) ? weights[r * org_width + c] : 0.0f;
            }
        }

        matmul_weights_const = op::Constant::create(element::Type_t::f32, Shape{ new_width, new_width }, new_weights);
    } else {
        matmul_weights_const = op::Constant::create(element::Type_t::f32, Shape{ new_width, new_width }, weights);
    }
    auto padded_input_shape = padded_input.get_shape();

    if (padded_input_shape[3] % 8 != 0 || padded_input_shape[0] != 1 || padded_input_shape[1] != 1)
        return nullptr;

    int conv_count = padded_input_shape[3] / 8;
    int len_after_split = (int)shape_size(padded_input_shape) / conv_count;
    auto leading_reshape_node = std::make_shared<ngraph::opset1::Reshape>(
        padded_input,
        op::Constant::create(ngraph::element::i64, Shape{ 2 }, { len_after_split, conv_count })->output(0),
        false);

    auto const_NC_to_CN = op::Constant::create(element::Type_t::i64, Shape{ 2 }, { 1ll, 0ll });
    auto const_CN_to_NC = op::Constant::create(element::Type_t::i64, Shape{ 2 }, { 1ll, 0ll });

    std::shared_ptr<Node> leading_permute = std::make_shared<op::Transpose>(leading_reshape_node->output(0),
        const_NC_to_CN->output(0));

    std::vector<int64_t> slits;
    for (int i = 0; i < conv_count; i++)
        slits.push_back(1);
    auto split = std::make_shared<op::VariadicSplit>(
        leading_permute->output(0),
        op::Constant::create(ngraph::element::i64, Shape{ 1 }, { 0 })->output(0),
        op::Constant::create(ngraph::element::i64, Shape{ 2 }, slits)->output(0));

    OutputVector chunks;

    for (int ci = 0; ci < conv_count; ci++) {
        auto input_to_matmul = std::make_shared<ngraph::opset1::Reshape>(
            split->output(ci),
            op::Constant::create(ngraph::element::i64, Shape{ 4 }, { 1,1,len_after_split / 8, 8 })->output(0),
            false);

        std::shared_ptr<Node> leading_conv_permute = std::make_shared<op::Transpose>(input_to_matmul->output(0),
                op::Constant::create(element::Type_t::i64, Shape{ 4 }, { 0, 3, 1, 2 }));

        auto conv_as_matmul = std::make_shared <op::MatMul>(leading_conv_permute, matmul_weights_const,
            false, true);
        conv_as_matmul->set_friendly_name(conv_name + "/" + std::to_string(ci));

        std::shared_ptr<Node> trailing_conv_permute = std::make_shared<op::Transpose>(conv_as_matmul->output(0),
            op::Constant::create(element::Type_t::i64, Shape{ 4 }, { 0, 2, 3, 1 }));

        auto output_from_matmul = std::make_shared<ngraph::opset1::Reshape>(
            trailing_conv_permute->output(0),
            op::Constant::create(ngraph::element::i64, Shape{ 2 }, { 1, len_after_split })->output(0),
            false);
        chunks.push_back(output_from_matmul->output(0));
    }
    auto concat = std::make_shared<opset1::Concat>(chunks, 0);
    std::shared_ptr<Node> trailing_permute = std::make_shared<op::Transpose>(concat->output(0),
        const_NC_to_CN->output(0));

    std::shared_ptr<Node> trailing_reshape = std::make_shared<ngraph::opset1::Reshape>(trailing_permute->output(0),
        op::Constant::create(ngraph::element::i64, Shape{ 4 }, padded_input_shape)->output(0),
        false);
    if (input_shape[2] & 7) {
        std::vector<size_t> split_sizes = { input_shape[2], pad_size };
        auto variadic_split = std::make_shared<ngraph::opset1::VariadicSplit>(
            trailing_reshape->output(0),
            op::Constant::create(ngraph::element::i64, Shape{ 1 }, { 2 })->output(0),
            op::Constant::create(ngraph::element::i64, Shape{ split_sizes.size() }, split_sizes)->output(0));
        return variadic_split;
    } else {
        return trailing_reshape;
    }
}

std::shared_ptr<Node> CreatePointwiseConvNHWC2NHCWBatchNormPrelu(
    const Output<Node>& input,
    std::string conv_name,
    AF af,
    ParameterProvider* provider)
{
    auto mul_mean_weights = provider->GetParametersAsFlatBufferByName(conv_name + "/mul_mean");
    auto prelu_slope = provider->GetParametersAsFlatBufferByName(conv_name + "/prelu");

    auto conv = CreatePointwiseConvNHWC2NHCWUsingMatMul(input, conv_name, provider);

    std::shared_ptr<Node> mul_mean_const = op::Constant::create(
        element::Type_t::f32, { 1,1,1,mul_mean_weights.size() }, mul_mean_weights);
    std::shared_ptr<Node> mul_mean = std::make_shared<op::v1::Multiply>(
        conv->output(0),
        mul_mean_const->output(0));
    mul_mean->set_friendly_name(conv_name + "/mul_mean");

    std::shared_ptr<Node> af_inst;
    if (af != AF_NONE) {
        std::shared_ptr<Node> af_inst;
        if (af == AF_RELU) {
            af_inst = std::make_shared<op::Relu>(mul_mean->output(0));
            af_inst->set_friendly_name(conv_name + "/relu");
        }
        else if (af == AF_PRELU) {
            auto prelu_slope_const = op::Constant::create(
                element::Type_t::f32, { 1 }, prelu_slope);
            af_inst = std::make_shared<op::v0::PRelu>(mul_mean->output(0), prelu_slope_const);
            af_inst->set_friendly_name(conv_name + "/prelu");
        }
        else if (af == AF_SIGMOID) {
            af_inst = std::make_shared<op::v0::Sigmoid>(mul_mean->output(0));
            af_inst->set_friendly_name(conv_name + "/sigmoid");
        }
        return af_inst;
    }

    return mul_mean;
}

std::shared_ptr<Node> CreateDolbyBlock(OutputVector inputs, int block_index, size_t feature_count, SinkVector& sinks, bool last, ParameterProvider* provider)
{
    // A. conv->bias->prelu
    // B. stream shift buffer
    // C. conv->bias=>mul(channels(0..16), sigmoid(channels(16,31)))
    // D. conv(concat(C,input))
    // E. T/conv/T->multiply-> prelu
    // F. conv(concat(E,C,input))->bias->prelu
    // G. stream shift buffer
    // H. conv->bias=>mul(channels(0..16), sigmoid(channels(16,31)))
    // I. conv(concat(H,E,C,input))->bias->prelu
    // J. stream shift buffer
    // K. conv->bias=>mul(channels(0..16), sigmoid(channels(16,31)))
    // L. conv(concat(K,H,E,C,input))->bias->prelu

    auto conv_A = CreateConvolutionAfterChannelWiseConcatAddBiasNActivationFunction(inputs, { 0,0 }, { 0,0 }, { 1,1 }, { 32,inputs.size()*16,1,1 }, "Conv_A_"+std::to_string(block_index),
        AF_PRELU, provider);

    auto stream_buffer_B = CreateStreamingBufferConcat(conv_A->output(0), { 1,2,feature_count,32 }, 1, "Stream_Buffer_B_" + std::to_string(block_index), sinks);

    auto conv_C = CreateWrappedConvolutionAddSigmoidVariadicSplitMul(stream_buffer_B->output(0),
        { 0,1 }, { 0,1 }, { 1, 1 }, { 32, 32, 3, 3 }, "Conv_C_" + std::to_string(block_index), provider);

    auto concat_D = OutputVector{ conv_C->output(0) };
    concat_D.insert(concat_D.end(), inputs.begin(), inputs.end());

    auto conv_D = CreateConvolutionAfterChannelWiseConcatAddBiasNActivationFunction(
        concat_D,
        { 0,0 }, { 0,0 }, { 1,1 }, { 16,16 * concat_D.size(),1,1 }, "Conv_D_" + std::to_string(block_index),
        AF_PRELU, provider);

    auto conv_E = CreatePointwiseConvNHWC2NHCWBatchNormPrelu(
        conv_D->output(0),
        "Conv_E_" + std::to_string(block_index),
        AF_PRELU, provider);

    auto concat_F = OutputVector{ conv_E->output(0), conv_C->output(0)};
    concat_F.insert(concat_F.end(), inputs.begin(), inputs.end());

    auto conv_F = CreateConvolutionAfterChannelWiseConcatAddBiasNActivationFunction(
        concat_F,
        { 0,0 }, { 0,0 }, { 1,1 }, { 32,16*concat_F.size(),1,1 }, "Conv_F_" + std::to_string(block_index),
        AF_PRELU, provider);

    auto stream_buffer_G =
        CreateStreamingBufferConcat(conv_F->output(0), { 1,6,feature_count,32 }, 1, "Stream_Buffer_G_"+ std::to_string(block_index), sinks);

    auto conv_H = CreateWrappedConvolutionAddSigmoidVariadicSplitMul(stream_buffer_G->output(0),
        { 0,1 }, { 0,1 }, { 3,1 }, { 32, 32, 3, 3 }, "Conv_H_" + std::to_string(block_index), provider);

    auto concat_I = OutputVector{ conv_H->output(0), conv_E->output(0), conv_C->output(0) };
    concat_I.insert(concat_I.end(), inputs.begin(), inputs.end());

    auto conv_I = CreateConvolutionAfterChannelWiseConcatAddBiasNActivationFunction(
        concat_I,
        { 0,0 }, { 0,0 }, { 1,1 }, { 32,16* concat_I.size(),1,1 }, "Conv_I_" + std::to_string(block_index),
        AF_PRELU, provider);

    auto stream_buffer_J =
        CreateStreamingBufferConcat(conv_I->output(0), { 1,18,feature_count,32 }, 1, "Stream_Buffer_J_" + std::to_string(block_index), sinks);

    auto conv_K = CreateWrappedConvolutionAddSigmoidVariadicSplitMul(stream_buffer_J->output(0),
        { 0,1 }, { 0,1 }, { 9,1 }, { 32, 32, 3, 3 }, "Conv_K_" + std::to_string(block_index), provider);

    auto concat_L = OutputVector{ conv_K->output(0), conv_H->output(0), conv_E->output(0), conv_C->output(0) };
    concat_L.insert(concat_L.end(), inputs.begin(), inputs.end());

    auto conv_L = CreateConvolutionAfterChannelWiseConcatAddBiasNActivationFunction(
        concat_L,
        { 0,0 }, { 0,0 }, { 1,1 }, (last ? Shape{ 1, concat_L.size() * 16, 1, 1 } : Shape{ 16, concat_L.size() * 16, 1, 1 }), "Conv_L_" + std::to_string(block_index),
        last ? AF_SIGMOID : AF_PRELU, provider);

    return conv_L;
}

std::shared_ptr<Function> createNgraphFunctionDolby(ParameterProvider* provider)
{
    SinkVector sinks;
    auto paramNode = std::make_shared<op::Parameter>(
        element::Type_t::f32, Shape(std::vector<size_t>{ {1, 64}}));
    paramNode->set_friendly_name("Parameter");
    // ------- reshape to 1x1x56x1 NHWC--------
    auto reshape_node = std::make_shared<ngraph::opset1::Reshape>(
        paramNode->output(0),
        op::Constant::create(ngraph::element::i64, Shape{ 4 }, { 1, 1, 64, 1 })->output(0),
        false);

    // ------- multiply -------
    std::vector<float> first_multiply_weights = provider->GetParametersAsFlatBufferByName("BatchNormalization_7/mean/Fused_Mul_");
    auto multiply_A_init = std::make_shared<ngraph::opset1::Multiply>(
        reshape_node->output(0),
        op::Constant::create(ngraph::element::f32, Shape{ 1, 1, 64, 1 }, first_multiply_weights)->output(0));

    // --------- streaming buffer --------
    auto stream_buffer_B_init = CreateStreamingBufferConcat(multiply_A_init->output(0), { 1,4,64,1 }, 1, "Stream_Buffer_B_Init", sinks);

    // --------- convolution ------------
    auto conv_C_init = CreateConvolutionAfterChannelWiseConcatAddBiasNActivationFunction(
        { stream_buffer_B_init->output(0) }, { 0,0 }, { 0,0 }, { 1,1 }, { 16, 1, 5, 3 }, "Conv_C_Init",
        AF_PRELU, provider);

    std::vector<int64_t> split_row_sizes { 56,6 };
    auto left_56_columns = std::make_shared<opset1::VariadicSplit>(conv_C_init,
        op::Constant::create(ngraph::element::i64, Shape{ 1 }, { 2ull }), // axis W
        op::Constant::create(ngraph::element::i64, Shape{ 2 }, split_row_sizes));


    auto processing_block_1 = CreateDolbyBlock({ left_56_columns->output(0) }, 1, 56, sinks, false, provider);
    auto downsample_56_28_op = Downsample2xW_NHWC_C16(processing_block_1->output(0), "GroupedConv_1", provider);
    auto processing_block_2 = CreateDolbyBlock({ downsample_56_28_op->output(0) }, 2, 28, sinks, false, provider);
    auto downsample_28_14_op = Downsample2xW_NHWC_C16(processing_block_2->output(0), "GroupedConv_2", provider);
    auto processing_block_3 = CreateDolbyBlock({ downsample_28_14_op->output(0) }, 3, 14, sinks, false, provider);
    auto downsample_14_7_op = Downsample2xW_NHWC_C16(processing_block_3->output(0), "GroupedConv_3", provider);
    auto processing_block_4 = CreateDolbyBlock({ downsample_14_7_op->output(0) }, 4, 7, sinks, false, provider);

    auto upsample_7_14_op = std::shared_ptr<Node>(Upsample2xW_NHWC(processing_block_4, "ConvTranspose_1", provider));
    auto processing_block_5 = CreateDolbyBlock({ upsample_7_14_op->output(0), processing_block_3->output(0) }, 5, 14, sinks, false, provider);
    auto upsample_14_28_op = std::shared_ptr<Node>(Upsample2xW_NHWC(processing_block_5, "ConvTranspose_2", provider));
    auto processing_block_6 = CreateDolbyBlock({ upsample_14_28_op->output(0), processing_block_2->output(0) }, 6, 28, sinks, false, provider);
    auto upsample_28_56_op = std::shared_ptr<Node>(Upsample2xW_NHWC(processing_block_6, "ConvTranspose_3", provider));
    auto processing_block_7 = CreateDolbyBlock({ upsample_28_56_op->output(0), processing_block_1->output(0) }, 7, 56, sinks, true, provider);

    auto result_full = std::make_shared<op::Result>(processing_block_7->output(0));
    std::shared_ptr<ngraph::Function> fnPtr = std::make_shared<ngraph::Function>(
        result_full, ngraph::ParameterVector{ paramNode }, "dolby_0_3");
    fnPtr->add_sinks(sinks);
    fnPtr->add_results({ result_full });

    return fnPtr;

}

class SimpleParametersProvider : public ParameterProvider {
public:
    SimpleParametersProvider(const char* filename)
        : ParameterProvider()
        , m_data{ nullptr }
        , m_filename{ filename }
        , m_filesize{ 0ull } {
        std::ifstream inputFile;

        std::ifstream weightFile(filename, std::ifstream::ate | std::ifstream::binary);
        int64_t fileSize = weightFile.tellg();

        if (fileSize < 0) {
            throw std::logic_error("Incorrect parameters file");
        }

        m_filesize = static_cast<size_t>(fileSize);
        m_data = new uint8_t[m_filesize];
        inputFile.open(filename, std::ios::binary | std::ios::in);
        if (!inputFile.is_open()) {
            throw std::logic_error("Cannot open parameters file");
        }

        if (!inputFile.read(reinterpret_cast<char*>(m_data), m_filesize)) {
            inputFile.close();
            throw std::logic_error("Cannot read bytes from parameters file");
        }

        inputFile.close();
    }

    virtual ~SimpleParametersProvider() {
        delete[]m_data;
    }

    void RegisterBuffer(std::string name, size_t offset, size_t size, Shape shape)
    {
        buffer_map[name] = DataLocation{ offset , size, shape };
    }
    virtual std::vector<float>GetParametersAsFlatBufferByName(std::string name)
    {
        if (m_data && buffer_map.count(name)) {
            DataLocation &dl = buffer_map[name];
            if (dl.offset + dl.size <= m_filesize) {
                float* begin = (float*)(m_data + dl.offset);
                float* end = begin + dl.size/sizeof(float);
                if (name == "BatchNormalization_7/mean/Fused_Mul_") {
                    std::vector<float> flat_buffer(1);
                    auto right_padding = std::vector<float>(7);
                    flat_buffer.insert(flat_buffer.end(), begin, end);
                    flat_buffer.insert(flat_buffer.end(), right_padding.begin(), right_padding.end());
                    return flat_buffer;
                } else {
                    return std::vector<float>(begin, end);
                }
            } else {
                std::cerr << "ERROR: buffer " << name << " location is out of file size" << std::endl;
                return std::vector<float>();
            }
        } else {
            std::cerr << "ERROR: buffer " << name << " not found!" << std::endl;
            return std::vector<float>();
        }
    }
protected:
    typedef struct {
        size_t offset;
        size_t size;
        Shape  shape;
    } DataLocation;

    std::unordered_map<std::string, DataLocation> buffer_map;
    uint8_t*    m_data;
    std::string m_filename;
    size_t      m_filesize;
};

int main(int argc, char* argv[]) {
    std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

    // --------------------------- 1. Load inference engine -------------------------------------
    std::cout << "Loading Inference Engine" << std::endl;
    Core ie;
    if (argc != 2)
    {
        std::cerr << "ERROR: please provide filename with model weights" << std::endl;
        return -1;
    }
    SimpleParametersProvider provider(argv[1]);

    provider.RegisterBuffer("BatchNormalization_7/mean/Fused_Mul_", 172, 224, { 1,1,56,1 });
    provider.RegisterBuffer("Conv_C_Init", 404, 960, { 16,1,3,5 });
    provider.RegisterBuffer("Conv_C_Init/bias", 1364, 64, { 1,16,1,1 });
    provider.RegisterBuffer("Conv_C_Init/prelu", 1428, 4, { 1,1,1 });
    provider.RegisterBuffer("Conv_A_1", 1432, 2048, { 32,16,1,1 });
    provider.RegisterBuffer("Conv_A_1/bias", 3480, 128, { 1, 32, 1, 1 });
    provider.RegisterBuffer("Conv_A_1/prelu", 1428, 4,{1,1,1});
    provider.RegisterBuffer("Conv_C_1", 3608, 36864,{32,32,3,3});
    provider.RegisterBuffer("Conv_C_1/bias", 3480, 128,{1,32,1,1});
    provider.RegisterBuffer("Conv_D_1", 40488, 2048,{16,32,1,1});
    provider.RegisterBuffer("Conv_D_1/bias", 42536, 64,{1,16,1,1});
    provider.RegisterBuffer("Conv_D_1/prelu", 1428, 4,{1,1,1});
    provider.RegisterBuffer("Conv_E_1", 42632, 12544,{56,56,1,1});
    provider.RegisterBuffer("Conv_E_1/mul_mean", 55176, 64,{1,16,1,1});
    provider.RegisterBuffer("Conv_E_1/prelu", 1428, 4,{1,1,1});
    provider.RegisterBuffer("Conv_F_1", 55240, 6144,{32,48,1,1});
    provider.RegisterBuffer("Conv_F_1/bias", 3480, 128,{1,32,1,1});
    provider.RegisterBuffer("Conv_F_1/prelu", 1428, 4,{1,1,1});
    provider.RegisterBuffer("Conv_H_1", 61384, 36864,{32,32,3,3});
    provider.RegisterBuffer("Conv_H_1/bias", 3480, 128, {1,32,1,1});
    provider.RegisterBuffer("Conv_I_1", 98248, 8192,{32,64,1,1});
    provider.RegisterBuffer("Conv_I_1/bias", 3480, 128,{1,32,1,1});
    provider.RegisterBuffer("Conv_I_1/prelu", 1428, 4,{1,1,1});
    provider.RegisterBuffer("Conv_K_1", 106440, 36864,{32,32,3,3});
    provider.RegisterBuffer("Conv_K_1/bias", 3480, 128,{1,32,1,1});
    provider.RegisterBuffer("Conv_L_1", 143304, 5120,{16,80,1,1});
    provider.RegisterBuffer("Conv_L_1/bias", 42536, 64,{1,16,1,1});
    provider.RegisterBuffer("Conv_L_1/prelu", 1428, 4,{1,1,1});
    provider.RegisterBuffer("GroupedConv_1", 148424, 128,{16,1,1,2,1});
    provider.RegisterBuffer("Conv_A_2", 148552, 2048,{32,16,1,1});
    provider.RegisterBuffer("Conv_A_2/bias", 3480, 128,{1,32,1,1});
    provider.RegisterBuffer("Conv_A_2/prelu", 1428, 4,{1,1,1});
    provider.RegisterBuffer("Conv_C_2", 150600, 36864,{32,32,3,3});
    provider.RegisterBuffer("Conv_C_2/bias", 3480, 128,{1,32,1,1});
    provider.RegisterBuffer("Conv_D_2", 187464, 2048,{16,32,1,1});
    provider.RegisterBuffer("Conv_D_2/bias", 42536, 64,{1,16,1,1});
    provider.RegisterBuffer("Conv_D_2/prelu", 1428, 4,{1,1,1});
    provider.RegisterBuffer("Conv_E_2", 189512, 3136,{28,28,1,1});
    provider.RegisterBuffer("Conv_E_2/mul_mean", 55176, 64,{1,16,1,1});
    provider.RegisterBuffer("Conv_E_2/prelu", 1428, 4,{1,1,1});
    provider.RegisterBuffer("Conv_F_2", 192648, 6144,{32,48,1,1});
    provider.RegisterBuffer("Conv_F_2/bias", 3480, 128,{1,32,1,1});
    provider.RegisterBuffer("Conv_F_2/prelu", 1428, 4,{1,1,1});
    provider.RegisterBuffer("Conv_H_2", 198792, 36864,{32,32,3,3});
    provider.RegisterBuffer("Conv_H_2/bias", 3480, 128,{1,32,1,1});
    provider.RegisterBuffer("Conv_I_2", 235656, 8192,{32,64,1,1});
    provider.RegisterBuffer("Conv_I_2/bias", 3480, 128,{1,32,1,1});
    provider.RegisterBuffer("Conv_I_2/prelu", 1428, 4,{1,1,1});
    provider.RegisterBuffer("Conv_K_2", 243848, 36864,{32,32,3,3});
    provider.RegisterBuffer("Conv_K_2/bias", 3480, 128,{1,32,1,1});
    provider.RegisterBuffer("Conv_L_2", 280712, 5120,{16,80,1,1});
    provider.RegisterBuffer("Conv_L_2/bias", 42536, 64,{1,16,1,1});
    provider.RegisterBuffer("Conv_L_2/prelu", 1428, 4,{1,1,1});
    provider.RegisterBuffer("GroupedConv_2", 285832, 128,{16,1,1,2,1});
    provider.RegisterBuffer("Conv_A_3", 285960, 2048,{32,16,1,1});
    provider.RegisterBuffer("Conv_A_3/bias", 3480, 128,{1,32,1,1});
    provider.RegisterBuffer("Conv_A_3/prelu", 1428, 4,{1,1,1});
    provider.RegisterBuffer("Conv_C_3", 288008, 36864,{32,32,3,3});
    provider.RegisterBuffer("Conv_C_3/bias", 3480, 128,{1,32,1,1});
    provider.RegisterBuffer("Conv_D_3", 324872, 2048,{16,32,1,1});
    provider.RegisterBuffer("Conv_D_3/bias", 42536, 64,{1,16,1,1});
    provider.RegisterBuffer("Conv_D_3/prelu", 1428, 4,{1,1,1});
    provider.RegisterBuffer("Conv_E_3", 326920, 784,{14,14,1,1});
    provider.RegisterBuffer("Conv_E_3/mul_mean", 55176, 64,{1,16,1,1});
    provider.RegisterBuffer("Conv_E_3/prelu", 1428, 4,{1,1,1});
    provider.RegisterBuffer("Conv_F_3", 327704, 6144,{32,48,1,1});
    provider.RegisterBuffer("Conv_F_3/bias", 3480, 128,{1,32,1,1});
    provider.RegisterBuffer("Conv_F_3/prelu", 1428, 4,{1,1,1});
    provider.RegisterBuffer("Conv_H_3", 333848, 36864,{32,32,3,3});
    provider.RegisterBuffer("Conv_H_3/bias", 3480, 128,{1,32,1,1});
    provider.RegisterBuffer("Conv_I_3", 370712, 8192,{32,64,1,1});
    provider.RegisterBuffer("Conv_I_3/bias", 3480, 128,{1,32,1,1});
    provider.RegisterBuffer("Conv_I_3/prelu", 1428, 4,{1,1,1});
    provider.RegisterBuffer("Conv_K_3", 378904, 36864,{32,32,3,3});
    provider.RegisterBuffer("Conv_K_3/bias", 3480, 128,{1,32,1,1});
    provider.RegisterBuffer("Conv_L_3", 415768, 5120,{16,80,1,1});
    provider.RegisterBuffer("Conv_L_3/bias", 42536, 64,{1,16,1,1});
    provider.RegisterBuffer("Conv_L_3/prelu", 1428, 4,{1,1,1});
    provider.RegisterBuffer("GroupedConv_3", 420888, 128,{16,1,1,2,1});
    provider.RegisterBuffer("Conv_A_4", 421016, 2048,{32,16,1,1});
    provider.RegisterBuffer("Conv_A_4/bias", 3480, 128,{1,32,1,1});
    provider.RegisterBuffer("Conv_A_4/prelu", 1428, 4,{1,1,1});
    provider.RegisterBuffer("Conv_C_4", 423064, 36864,{32,32,3,3});
    provider.RegisterBuffer("Conv_C_4/bias", 3480, 128,{1,32,1,1});
    provider.RegisterBuffer("Conv_D_4", 459928, 2048,{16,32,1,1});
    provider.RegisterBuffer("Conv_D_4/bias", 42536, 64,{1,16,1,1});
    provider.RegisterBuffer("Conv_D_4/prelu", 1428, 4,{1,1,1});
    provider.RegisterBuffer("Conv_E_4", 461976, 196,{7,7,1,1});
    provider.RegisterBuffer("Conv_E_4/mul_mean", 55176, 64,{1,16,1,1});
    provider.RegisterBuffer("Conv_E_4/prelu", 1428, 4,{1,1,1});
    provider.RegisterBuffer("Conv_F_4", 462172, 6144,{32,48,1,1});
    provider.RegisterBuffer("Conv_F_4/bias", 3480, 128,{1,32,1,1});
    provider.RegisterBuffer("Conv_F_4/prelu", 1428, 4,{1,1,1});
    provider.RegisterBuffer("Conv_H_4", 468316, 36864,{32,32,3,3});
    provider.RegisterBuffer("Conv_H_4/bias", 3480, 128,{1,32,1,1});
    provider.RegisterBuffer("Conv_I_4", 505180, 8192,{32,64,1,1});
    provider.RegisterBuffer("Conv_I_4/bias", 3480, 128,{1,32,1,1});
    provider.RegisterBuffer("Conv_I_4/prelu", 1428, 4,{1,1,1});
    provider.RegisterBuffer("Conv_K_4", 513372, 36864,{32,32,3,3});
    provider.RegisterBuffer("Conv_K_4/bias", 3480, 128,{1,32,1,1});
    provider.RegisterBuffer("Conv_L_4", 550236, 5120,{16,80,1,1});
    provider.RegisterBuffer("Conv_L_4/bias", 42536, 64,{1,16,1,1});
    provider.RegisterBuffer("Conv_L_4/prelu", 1428, 4,{1,1,1});
    provider.RegisterBuffer("ConvTranspose_1", 555356, 2048,{16,16,2,1});
    provider.RegisterBuffer("ConvTranspose_1/bias", 557404, 64,{1,16,1,1});
    provider.RegisterBuffer("ConvTranspose_1/prelu", 1428, 4,{1,1,1});
    provider.RegisterBuffer("Conv_A_5", 557468, 4096,{32,32,1,1});
    provider.RegisterBuffer("Conv_A_5/bias", 3480, 128,{1,32,1,1});
    provider.RegisterBuffer("Conv_A_5/prelu", 1428, 4,{1,1,1});
    provider.RegisterBuffer("Conv_C_5", 561564, 36864,{32,32,3,3});
    provider.RegisterBuffer("Conv_C_5/bias", 3480, 128,{1,32,1,1});
    provider.RegisterBuffer("Conv_D_5", 598428, 3072,{16,48,1,1});
    provider.RegisterBuffer("Conv_D_5/bias", 42536, 64,{1,16,1,1});
    provider.RegisterBuffer("Conv_D_5/prelu", 1428, 4,{1,1,1});
    provider.RegisterBuffer("Conv_E_5", 601500, 784,{14,14,1,1});
    provider.RegisterBuffer("Conv_E_5/mul_mean", 55176, 64,{1,16,1,1});
    provider.RegisterBuffer("Conv_E_5/prelu", 1428, 4,{1,1,1});
    provider.RegisterBuffer("Conv_F_5", 602284, 8192,{32,64,1,1});
    provider.RegisterBuffer("Conv_F_5/bias", 3480, 128,{1,32,1,1});
    provider.RegisterBuffer("Conv_F_5/prelu", 1428, 4,{1,1,1});
    provider.RegisterBuffer("Conv_H_5", 610476, 36864,{32,32,3,3});
    provider.RegisterBuffer("Conv_H_5/bias", 3480, 128,{1,32,1,1});
    provider.RegisterBuffer("Conv_I_5", 647340, 10240,{32,80,1,1});
    provider.RegisterBuffer("Conv_I_5/bias", 3480, 128,{1,32,1,1});
    provider.RegisterBuffer("Conv_I_5/prelu", 1428, 4,{1,1,1});
    provider.RegisterBuffer("Conv_K_5", 657580, 36864,{32,32,3,3});
    provider.RegisterBuffer("Conv_K_5/bias", 3480, 128,{1,32,1,1});
    provider.RegisterBuffer("Conv_L_5", 694444, 6144,{16,96,1,1});
    provider.RegisterBuffer("Conv_L_5/bias", 42536, 64,{1,16,1,1});
    provider.RegisterBuffer("Conv_L_5/prelu", 1428, 4,{1,1,1});
    provider.RegisterBuffer("ConvTranspose_2", 700588, 2048,{16,16,2,1});
    provider.RegisterBuffer("ConvTranspose_2/bias", 702636, 64,{1,16,1,1});
    provider.RegisterBuffer("ConvTranspose_2", 1428, 4,{1,1,1});
    provider.RegisterBuffer("Conv_A_6", 702700, 4096,{32,32,1,1});
    provider.RegisterBuffer("Conv_A_6/bias", 3480, 128,{1,32,1,1});
    provider.RegisterBuffer("Conv_A_6/prelu", 1428, 4,{1,1,1});
    provider.RegisterBuffer("Conv_C_6", 706796, 36864,{32,32,3,3});
    provider.RegisterBuffer("Conv_C_6/bias", 3480, 128,{1,32,1,1});
    provider.RegisterBuffer("Conv_D_6", 743660, 3072,{16,48,1,1});
    provider.RegisterBuffer("Conv_D_6/bias", 42536, 64,{1,16,1,1});
    provider.RegisterBuffer("Conv_D_6/prelu", 1428, 4,{1,1,1});
    provider.RegisterBuffer("Conv_E_6", 746732, 3136,{28,28,1,1});
    provider.RegisterBuffer("Conv_E_6/mul_mean", 55176, 64,{1,16,1,1});
    provider.RegisterBuffer("Conv_E_6/prelu", 1428, 4,{1,1,1});
    provider.RegisterBuffer("Conv_F_6", 749868, 8192,{32,64,1,1});
    provider.RegisterBuffer("Conv_F_6/bias", 3480, 128,{1,32,1,1});
    provider.RegisterBuffer("Conv_F_6/prelu", 1428, 4,{1,1,1});
    provider.RegisterBuffer("Conv_H_6", 758060, 36864,{32,32,3,3});
    provider.RegisterBuffer("Conv_H_6/bias", 3480, 128,{1,32,1,1});
    provider.RegisterBuffer("Conv_I_6", 794924, 10240,{32,80,1,1});
    provider.RegisterBuffer("Conv_I_6/bias", 3480, 128,{1,32,1,1});
    provider.RegisterBuffer("Conv_I_6/prelu", 1428, 4,{1,1,1});
    provider.RegisterBuffer("Conv_K_6", 805164, 36864,{32,32,3,3});
    provider.RegisterBuffer("Conv_K_6/bias", 3480, 128,{1,32,1,1});
    provider.RegisterBuffer("Conv_L_6", 842028, 6144,{16,96,1,1});
    provider.RegisterBuffer("Conv_L_6/bias", 42536, 64,{1,16,1,1});
    provider.RegisterBuffer("Conv_L_6/prelu", 1428, 4,{1,1,1});
    provider.RegisterBuffer("ConvTranspose_3", 848172, 2048,{16,16,2,1});
    provider.RegisterBuffer("ConvTranspose_3/bias", 850220, 64,{1,16,1,1});
    provider.RegisterBuffer("ConvTranspose_3", 1428, 4,{1,1,1});
    provider.RegisterBuffer("Conv_A_7", 850284, 4096,{32,32,1,1});
    provider.RegisterBuffer("Conv_A_7/bias", 3480, 128,{1,32,1,1});
    provider.RegisterBuffer("Conv_A_7/prelu", 1428, 4,{1,1,1});
    provider.RegisterBuffer("Conv_C_7", 854380, 36864,{32,32,3,3});
    provider.RegisterBuffer("Conv_C_7/bias", 3480, 128,{1,32,1,1});
    provider.RegisterBuffer("Conv_D_7", 891244, 3072,{16,48,1,1});
    provider.RegisterBuffer("Conv_D_7/bias", 42536, 64,{1,16,1,1});
    provider.RegisterBuffer("Conv_D_7/prelu", 1428, 4,{1,1,1});
    provider.RegisterBuffer("Conv_E_7", 894316, 12544,{56,56,1,1});
    provider.RegisterBuffer("Conv_E_7/mul_mean", 55176, 64,{1,16,1,1});
    provider.RegisterBuffer("Conv_E_7/prelu", 1428, 4,{1,1,1});
    provider.RegisterBuffer("Conv_F_7", 906860, 8192,{32,64,1,1});
    provider.RegisterBuffer("Conv_F_7/bias", 3480, 128,{1,32,1,1});
    provider.RegisterBuffer("Conv_F_7/prelu", 1428, 4,{1,1,1});
    provider.RegisterBuffer("Conv_H_7", 915052, 36864,{32,32,3,3});
    provider.RegisterBuffer("Conv_H_7/bias", 3480, 128,{1,32,1,1});
    provider.RegisterBuffer("Conv_I_7", 951916, 10240,{32,80,1,1});
    provider.RegisterBuffer("Conv_I_7/bias", 3480, 128,{1,32,1,1});
    provider.RegisterBuffer("Conv_I_7/prelu", 1428, 4,{1,1,1});
    provider.RegisterBuffer("Conv_K_7", 962156, 36864,{32,32,3,3});
    provider.RegisterBuffer("Conv_K_7/bias", 3480, 128,{1,32,1,1});
    provider.RegisterBuffer("Conv_L_7", 999020, 384,{1,96,1,1});
    provider.RegisterBuffer("Conv_L_7/bias", 999404, 4,{1,1,1,1});
    //--------------------------- 2. Create network using ngraph function -----------------------------------
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::Serialize>("model.xml", "model.bin", ngraph::pass::Serialize::Version::IR_V10);
    manager.register_pass <ngraph::pass::Conv2dDecomposition>();
    manager.register_pass<ngraph::pass::Serialize>("model_factorized.xml", "model_factorized.bin", ngraph::pass::Serialize::Version::IR_V10);
    const auto& pass_config = manager.get_pass_config();
    manager.run_passes(createNgraphFunctionDolby(&provider));

    return 0;
}