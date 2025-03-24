from .QHNet_flow import QHNet_flow
from .QHNet_flow_v2 import QHNet_flow as QHNet_flow_v2
from .QHNet_flow_v3 import QHNet_flow as QHNet_flow_v3
from .QHNet_flow_v4 import QHNet_flow as QHNet_flow_v4
from .QHNet_flow_v5 import QHNet_flow as QHNet_flow_v5
from .QHNet_flow_v5_qh9 import QHNet_flow as QHNet_flow_v5_qh9
from .QHNet_flow_escn import QHNet_flow as QHNet_flow_escn

from .QHNet import QHNet
from .DEQHNet import DEQHNet
from .Real_QHNet import QHNet as Real_QHNet
from .Real_QHNet_qh9 import QHNet as Real_QHNet_qh9
from .QHNet_consistent import QHNet_consistent
from .QHNet_consistent_w_t import QHNet_consistent as QHNet_consistent_w_t
from .QHNet_consistent_w_t_discrete import (
    QHNet_consistent as QHNet_consistent_w_t_discrete,
)
from .QHNet_consistent_w_t_discrete_v2 import (
    QHNet_consistent as QHNet_consistent_w_t_discrete_v2,
)
from .QHNet_consistent_w_t_discrete_v3 import (
    QHNet_consistent as QHNet_consistent_w_t_discrete_v3,
)
from .QHNet_consistent_w_t_discrete_small import (
    QHNet_consistent as QHNet_consistent_w_t_discrete_small,
)
from .QHNet_consistent_refinement import QHNet_consistent_refinement
from .FrameNet import FrameNet
from .FrameNet_v2 import FrameNet_V2
from .FrameNet_v3 import FrameNet_V3

from .Real_QHNet_escn import QHNet as Real_QHNet_escn
from .Real_QHNet_light import QHNet as Real_QHNet_light

import logging

logger = logging.getLogger(__name__)

__all__ = ["get_model"]

# version: wo bias and with bias model are used to load the model for the paper reproduction
# QHNet is the clean version, and we use QHNet to build the benchmark performance


def get_model(args):
    model_args = {
        "in_node_features": getattr(args, "in_node_features", 1),
        "sh_lmax": getattr(args, "sh_lmax", 4),
        "hidden_size": getattr(args, "hidden_size", 128),
        "bottle_hidden_size": getattr(args, "bottle_hidden_size", 32),
        "num_gnn_layers": getattr(args, "num_gnn_layers", 5),
        "max_radius": getattr(args, "max_radius", 15),
        "num_nodes": getattr(args, "num_nodes", 10),
        "radius_embed_dim": getattr(args, "radius_embed_dim", 16),
        "max_T": getattr(args, "max_T", 15),
        "use_block_S": getattr(args, "use_block_S", False),
        "f_max_iter": 5,
        "b_max_iter": 5,
        "ham_dim": getattr(args, "ham_dim", 24),
        "ham_hidden": getattr(args, "ham_hidden", 24 * 24 // 2),
    }
    logging.info(f"model_args: {model_args}")
    if args.version.lower() == "Real_QHNet".lower():
        return Real_QHNet(**model_args)
    elif args.version.lower() == "Real_QHNet_qh9".lower():
        return Real_QHNet_qh9(**model_args)
    elif args.version.lower() == "Real_QHNet_escn".lower():
        return Real_QHNet_escn(**model_args)
    elif args.version.lower() == "Real_QHNet_light".lower():
        return Real_QHNet_light(**model_args)
    elif args.version.lower() == "QHNet".lower():
        return QHNet(**model_args)
    elif args.version.lower() == "DEQHNet".lower():
        return DEQHNet(**model_args)

    if args.version.lower() == "QHNet_flow".lower():
        return QHNet_flow(**model_args)
    elif args.version.lower() == "QHNet_flow_v2".lower():
        return QHNet_flow_v2(**model_args)
    elif args.version.lower() == "QHNet_flow_v3".lower():
        return QHNet_flow_v3(**model_args)
    elif args.version.lower() == "QHNet_flow_v4".lower():
        return QHNet_flow_v4(**model_args)
    elif args.version.lower() == "QHNet_flow_v5".lower():
        return QHNet_flow_v5(**model_args)
    elif args.version.lower() == "QHNet_flow_v5_qh9".lower():
        return QHNet_flow_v5_qh9(**model_args)

    if args.version.lower() == "QHNet_flow_escn".lower():
        return QHNet_flow_escn(**model_args)

    if args.version.lower() == "QHNet_consistent".lower():
        return QHNet_consistent(**model_args)
    elif args.version.lower() == "QHNet_consistent_w_t".lower():
        return QHNet_consistent_w_t(**model_args)
    elif args.version.lower() == "QHNet_consistent_w_t_discrete".lower():
        return QHNet_consistent_w_t_discrete(**model_args)
    elif args.version.lower() == "QHNet_consistent_w_t_discrete_v2".lower():
        return QHNet_consistent_w_t_discrete_v2(**model_args)
    elif args.version.lower() == "QHNet_consistent_w_t_discrete_v3".lower():
        return QHNet_consistent_w_t_discrete_v3(**model_args)
    elif args.version.lower() == "QHNet_consistent_w_t_discrete_small".lower():
        return QHNet_consistent_w_t_discrete_small(**model_args)
    elif args.version.lower() == "QHNet_consistent_refinement".lower():
        return QHNet_consistent_refinement(**model_args)

    if args.version.lower() == "FrameNet".lower():
        return FrameNet(**model_args)
    elif args.version.lower() == "FrameNet_v2".lower():
        return FrameNet_V2(**model_args)
    elif args.version.lower() == "FrameNet_v3".lower():
        return FrameNet_V3(**model_args)

    raise NotImplementedError(f"the version {args.version} is not implemented.")


from pl_module.base_module import LitModel
from pl_module.flow_module import LitModel_flow
from pl_module.flow_module_cf_loss import LitModel_flow_cf
from pl_module.consistent_module import LitModel_consistent
from pl_module.consistent_module_loss2 import (
    LitModel_consistent as LitModel_consistent_loss2,
)
from pl_module.consistent_module_t import LitModel_consistent_t as LitModel_consistent_t
from pl_module.consistent_module_t_v2 import (
    LitModel_consistent_v2 as LitModel_consistent_t_v2,
)
from pl_module.consistent_module_t_v3 import (
    LitModel_consistent_v3 as LitModel_consistent_t_v3,
)
from pl_module.consistent_module_refinement import LitModel_consistent_refinement
from pl_module.consistent_flow_module import LitModel_consistent_flow
from pl_module.flow_module_qh9 import LitModel_flow as LitModel_flow_qh9


def get_pl_model(conf):
    version = conf.model.version.lower()
    pl_type = conf.get("pl_type", None)

    if pl_type == "flow":
        return LitModel_flow
    elif pl_type == "flow_qh9":
        return LitModel_flow_qh9
    elif pl_type == "flow_cf":
        return LitModel_flow_cf
    elif pl_type == "consistent":
        return LitModel_consistent
    elif pl_type == "consistent_loss2":
        return LitModel_consistent_loss2
    elif pl_type == "consistent_t":
        return LitModel_consistent_t
    elif pl_type == "consistent_t_v2":
        return LitModel_consistent_t_v2
    elif pl_type == "consistent_t_v3":
        return LitModel_consistent_t_v3
    elif pl_type == "consistent_flow":
        return LitModel_consistent_flow
    elif pl_type == "base":
        return LitModel

    if "flow" in version:
        return LitModel_flow
    elif "frame".lower() in version.lower():
        return LitModel_flow
    elif "refine" in version:
        return LitModel_consistent_refinement
    elif "w_t" in version:
        if "v2" in version:
            return LitModel_consistent_t_v2
        if "v3" in version:
            return LitModel_consistent_t_v3
        return LitModel_consistent_t
    elif "consistency" in version or "consistent" in version:
        return LitModel_consistent
    else:
        return LitModel
