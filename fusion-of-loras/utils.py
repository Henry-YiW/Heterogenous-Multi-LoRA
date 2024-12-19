import torch

def reverse_token_mapping(text_indexes, offset_mappings):
    matched_mapping_indexes = []
    for index, mapping in enumerate(offset_mappings):
        if mapping[0] <= text_indexes[0] and mapping[1] >= text_indexes[1]:
            matched_mapping_indexes.append(index)
    return matched_mapping_indexes


def tensor_insert(tensor, value, index, dim):
    # Ensure value has the correct number of dimensions
    if tensor.dim() != value.dim():
        raise ValueError("Value must have the same number of dimensions as the input tensor.")

    # Split the tensor along the specified dimension
    if index > tensor.size(dim):
        raise IndexError(f"Index {index} is out of bounds for dimension {dim} with size {tensor.size(dim)}")

    # Slice and concatenate
    before = tensor.narrow(dim, 0, index)  # Select everything before the index
    after = tensor.narrow(dim, index, tensor.size(dim) - index)  # Select everything after the index
    return torch.cat([before, value, after], dim=dim)
    