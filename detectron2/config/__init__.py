# Copyright (c) Facebook, Inc. and its affiliates.
from .compat import downgrade_config, upgrade_config
from .config import CfgNode, get_cfg, global_cfg, set_global_cfg, configurable
from .config_hrnet import add_hrnet_config
from .instantiate import instantiate
from .lazy import LazyCall, LazyConfig

__all__ = [
    "CfgNode",
    "get_cfg",
    "global_cfg",
    "set_global_cfg",
    "downgrade_config",
    "upgrade_config",
    "configurable",
    "instantiate",
    "LazyCall",
    "LazyConfig",
    "add_hrnet_config",
]


from detectron2.utils.env import fixup_module_metadata

fixup_module_metadata(__name__, globals(), __all__)
del fixup_module_metadata
