# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from openvino.tools.mo.front.extractor import get_new_placeholder_name
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.utils import refer_to_faq_msg

# pylint: disable=no-name-in-module,import-error
from openvino.tools.ovc.convert_data_type import packed_U1, packed_U4, packed_I4, SUPPORTED_DATA_TYPES, \
    data_type_str_to_np, data_type_str_to_precision, data_type_str_to_destination_type, np_data_type_to_precision, \
    np_data_type_to_destination_type, destination_type_to_np_data_type, precision_to_destination_type


def convert_blob(blob: np.ndarray, dst_type: type):
    if blob.dtype == dst_type:
        return blob, None, None

    converted_blob = blob.astype(dtype=dst_type, casting="unsafe")
    if dst_type in (np.int32, np.int64, np.uint8, np.int8) and not np.array_equal(blob, converted_blob):
        raise Error('The conversion of blob with value "{}" to dst_type "{}" results in rounding'.format(
            blob, dst_type))

    finite_match = (np.isfinite(blob) != np.isfinite(converted_blob))
    zero_match = ((blob == 0) != (converted_blob == 0))
    finite_match_count = np.count_nonzero(finite_match)
    zero_match_count = np.count_nonzero(zero_match)

    return converted_blob, finite_match_count, zero_match_count


def convert_node_blobs(graph: Graph, node: Node, data_type: type):
    out_edges = graph.out_edges(node.node, data=True)

    # if the data.value is used as binary weights
    if any('bin' in d for _, __, d in out_edges):
        blob = node.value
        if blob.dtype != data_type:
            new_blob, finite_match_count, zero_match_count = convert_blob(blob, data_type)
            consumers = [x.name if x.has_valid('name') else '<NO NAME>' for x in node.out_nodes()]
            log.debug(
                'Blob was converted to {} while dumping to the bin file. This blob is an input for {} nodes.'.format(
                    data_type, consumers))
            if finite_match_count:
                log.error(
                    ("{} elements of {} were clipped to infinity while converting a blob for node [{}] to {}. " +
                     refer_to_faq_msg(76)).format(finite_match_count, blob.size, consumers, data_type))
            if zero_match_count:
                log.warning(
                    ("{} elements of {} were clipped to zero while converting a blob for node [{}] to {}. " +
                     refer_to_faq_msg(77)).format(zero_match_count, blob.size, consumers, data_type))

            node.value = new_blob
            # for the constant node need to propagate the converted value to the node output because there is a fake
            # input data for the 'Const' nodes being generated in the CreateConstNodesReplacement
            if len(node.out_nodes()) == 1 and node.out_node(0).op == 'Const':
                const_node = node.out_node(0)
                const_node.value = new_blob
                const_node.infer(const_node)
                const_node.type_infer(const_node)


def convert_parameters_data_type(graph: Graph, data_type_str: str):
    inputs = graph.get_op_nodes(op='Parameter')
    data_type = data_type_str_to_np(data_type_str)
    user_defined_data_types = graph.graph['user_shapes'] if 'user_shapes' in graph.graph else None
    for input in inputs:
        user_defined_type = None
        name = input.soft_get('initial_node_name', input.id)

        # override data type for Parameter specified by the user. This is a workaround for the issue in the
        # extensions.middle.ChangePlaceholderTypes transformation which has an incorrect condition and always overrides
        # Parameter data type to np.float32. When the transformation is fixed the code below must be updated
        if user_defined_data_types is not None and name in user_defined_data_types:
            for desc in user_defined_data_types[name]:
                if 'port' in desc and desc['port'] is None:  # neither input nor output port specified
                    user_defined_type = desc.get('data_type', None)
                else:  # need to check the particular port the Parameter was created for
                    p_name = get_new_placeholder_name(name, 'out' in desc, desc['out'] if 'out' in desc else desc['in'])
                    if p_name == input.soft_get('name'):
                        user_defined_type = desc.get('data_type', None)
        if user_defined_type is not None:
            log.info('Overriding Parameter node {} data type to {}'.format(name, user_defined_type))
            input['data_type'] = user_defined_type
            input.out_port(0).set_data_type(user_defined_type, True)
        elif not input.has_valid('data_type') or input.data_type == np.float32:
            input['data_type'] = data_type
            input.out_port(0).set_data_type(data_type, True)
        else:
            log.info('Do not change data type for node {}'.format(input.soft_get('name')))


def convert_blobs(graph: Graph, data_type_str: str):
    for node in graph.get_data_nodes():
        if node.value is not None:
            try:
                if node.value.dtype in [np.float32, np.float64, np.float16] and not node.has_and_set('correct_data_type'):
                    convert_node_blobs(graph, node, data_type_str_to_np(data_type_str))
            except Exception as e:
                raise Error('Coudn\'t convert blob {}, details: {}', node.soft_get('name'), e) from e
