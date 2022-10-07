import logging
import os

logger = logging.getLogger(__name__)


def get_input_value(input_docs, inputs, operator):
    input_value = None
    if inputs is not None and len(inputs) > 0:
        src_nm_0 = inputs[0].get('source', '')
        fp_0 = inputs[0].get('field_path', '')
        iv_0 = get_value_from_path(input_docs[src_nm_0], src_nm_0, fp_0)
        if len(inputs) == 1:
            input_value = iv_0
        elif len(inputs) == 2 and operator is not None:
            src_nm_1 = inputs[1].get('source', '')
            fp_1 = inputs[1].get('field_path', '')
            iv_1 = get_value_from_path(input_docs[src_nm_1], src_nm_1, fp_1)
            if operator == 'division':
                input_value = iv_0 / iv_1
            else:
                logger.warning("Arithmetic operations other than division are not supported yet.")
        else:
            logger.warning("More than 2 inputs are not supported yet.")
    else:
        logger.warning("inputs are null")
    return input_value


def get_value_from_path(input_doc, source_name, path):
    if input_doc is None:
        logger.warning(f"{source_name} is null")
        return None
    else:
        d = input_doc
        for nested_key in path.split('/'):
            temp = d.get(nested_key)
            if isinstance(temp, dict):
                d = d.get(nested_key)
            else:
                return temp
    return None


def set_file(file):
    if not os.path.exists(file):
        with open(file, 'a+') as f:
            pass