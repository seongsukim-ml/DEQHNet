from .DEQHNet import DEQHNet


__all__ = ['get_model']

# version: wo bias and with bias model are used to load the model for the paper reproduction
# QHNet is the clean version, and we use QHNet to build the benchmark performance

def get_model(args):
    if args.model.lower() == 'DEQHNet'.lower():
        return DEQHNet(
            in_node_features=1,
            sh_lmax=4,
            hidden_size=128,
            bottle_hidden_size=32,
            num_gnn_layers=5,
            max_radius=15,
            num_nodes=10,
            radius_embed_dim=16,
            f_max_iter=5,
            b_max_iter=5
        )
    else:
        raise NotImplementedError(
            f"the model {args.model} is not implemented.")
