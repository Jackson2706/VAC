# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from fvcore.common.registry import Registry

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for datasets.

The registered object will be called with `obj(cfg, split)`.
The call should return a `torch.utils_.data.Dataset` object.
"""


def build_dataset(dataset_name, cfg, split):
    """
    Build a datasets, defined by `dataset_name`.
    Args:
        dataset_name (str): the name of the datasets to be constructed.
        cfg (CfgNode): configs. Details can be found in
            slowfast/configs/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    Returns:
        Dataset: a constructed datasets specified by dataset_name.
    """
    # Capitalize the the first letter of the dataset_name since the dataset_name
    # in configs may be in lowercase but the name of datasets class should always
    # start with an uppercase letter.
    name = dataset_name.capitalize()
    return DATASET_REGISTRY.get(name)(cfg, split)
