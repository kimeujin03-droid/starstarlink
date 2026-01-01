import torch


def load_ckpt_strip_orig_mod(model, ckpt_path, map_location="cpu", strict=True, verbose=True):
    """Load a checkpoint that may have `_orig_mod.` prefixes in keys.

    This happens frequently when saving a model wrapped/compiled by torch.compile.

    Parameters
    ----------
    model : torch.nn.Module
    ckpt_path : str
    map_location : str
    strict : bool
    verbose : bool

    Returns
    -------
    model : torch.nn.Module
    """
    ckpt = torch.load(ckpt_path, map_location=map_location)

    state = None
    if isinstance(ckpt, dict):
        # Try common keys
        for k in ("model", "state_dict", "net", "weights"):
            if k in ckpt and isinstance(ckpt[k], dict):
                state = ckpt[k]
                break

    if state is None:
        # Assume raw state_dict
        if isinstance(ckpt, dict):
            state = ckpt
        else:
            raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")

    new_state = {}
    for k, v in state.items():
        if k.startswith("_orig_mod."):
            new_state[k[len("_orig_mod."):]] = v
        else:
            new_state[k] = v

    missing, unexpected = model.load_state_dict(new_state, strict=strict)

    if verbose:
        print(f"[ckpt] Loaded: {ckpt_path}")
        print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}")
        if len(missing) and len(missing) <= 10:
            print("  missing:", missing)
        if len(unexpected) and len(unexpected) <= 10:
            print("  unexpected:", unexpected)

    return model
