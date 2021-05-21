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
#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset2.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/pass/pass.hpp"
#include "transformations_visibility.hpp"

#include "transformations/serialize.hpp"

using namespace InferenceEngine;
using namespace ngraph;
using namespace op;

namespace ngraph {
namespace pass {

class AddMemoryCells;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::AddMemoryCells : public ngraph::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;

    bool run_on_function(std::shared_ptr<ngraph::Function> f) override {
        int memory_cell_id = 1;
        //bool is_graph_modfied = false;
        for (auto& node : f->get_ordered_ops()) {
            auto input = std::dynamic_pointer_cast<op::Parameter>(node);
            if (!input) continue;

            auto input_connected_to_list = input->get_output_target_inputs(0);

            if (input_connected_to_list.size() == 1) {
                auto next_node_to_input = input_connected_to_list.begin()->get_node()->shared_from_this();
                auto crop = std::dynamic_pointer_cast<ngraph::opset1::StridedSlice>(next_node_to_input);
                if (crop) {
                    auto crop_connected_to_list = crop->get_output_target_inputs(0);
                    auto next_node_to_crop = crop_connected_to_list.begin()->get_node()->shared_from_this();
                    auto concat = std::dynamic_pointer_cast<ngraph::opset1::Concat>(next_node_to_crop);
                    if (concat) {
                        printf("Concat with memory cell: %s\n", concat->get_friendly_name().c_str());
                        size_t flat_memory_size = shape_size(input->get_shape());
                        std::vector<float> initial_memory_state_data(flat_memory_size);
                        std::string memory_cell_name("MemoryCell_");
                        memory_cell_name += std::to_string(memory_cell_id);

                        std::shared_ptr<Node> initial_memory_state_const = op::Constant::create(
                            element::Type_t::f32, input->get_shape(), initial_memory_state_data);

                        auto read_value_node = std::make_shared<ngraph::opset3::ReadValue>(
                            initial_memory_state_const,
                            memory_cell_name);
                        ngraph::replace_node(input, read_value_node);
                        f->remove_parameter(input);

                        auto store_value_node = std::make_shared<ngraph::opset3::Assign>(
                            concat->output(0),
                            memory_cell_name);
                        auto concat_connected_to_list = concat->get_output_target_inputs(0);
                        for (auto co : concat_connected_to_list) {
                            auto node = co.get_node()->shared_from_this();
                            if (is_type<op::Result>(node)) {
                                auto result = std::dynamic_pointer_cast<op::Result>(node);
                                f->remove_result(result);
                                break;
                            }
                        }
                        f->add_sinks({ store_value_node });
                        memory_cell_id++;
                    }
                }
            }
        }
        return memory_cell_id != 1;
    }
};

NGRAPH_RTTI_DEFINITION(ngraph::pass::AddMemoryCells, "AddMemoryCells", 0);

int main(int argc, char* argv[]) {
    std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

    // --------------------------- 1. Load inference engine -------------------------------------
    std::cout << "Loading Inference Engine" << std::endl;
    Core ie;

    CNNNetwork net = ie.ReadNetwork(argv[1]);

    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::AddMemoryCells>();
    manager.register_pass<ngraph::pass::Serialize>("model_w_memory_cells.xml",
        "model_w_memory_cells.bin", ngraph::pass::Serialize::Version::IR_V10);
    manager.run_passes(net.getFunction());

    return 0;
}
