# TODO: Maybe move all of this elsewhere


def get_config(
    dataset: str,
    backbone: str,
    group: str,
    input_action: str,
    output_action: str,
    gamma: str,
) -> dict:
    return {
        "dataset": dataset,
        "backbone": backbone,
        "gamma": gamma,
        "group": group,
        "input_action": input_action,
        "output_action": output_action,
        # Default parameters that can be overridden with --config
        "dim": 4,
        "base_hidden_channels": [250, 250],
        "gamma_hidden_channels": [250],
        "loss": "relative-sse",
        "num_test_examples": 1000,
        "num_train_examples": 10000,
        "batch_size": 100,  # 10000/100=100 gradient steps per epoch
        "num_epochs": 1000,
        "learning_rate": 1e-4,
        "num_train_samples": 10,
        "num_test_samples": 100,
        "epochs_per_test": 25,
    }


def apply_config_overrides(
    config: dict, overrides: dict, seed: int, device: str
) -> dict:
    for k in overrides:
        if k not in config:
            raise IndexError(f"Trying to override an invalid config item: {k}")

        elif k in ["seed", "device"]:
            raise ValueError(f"Cannot override config item: {k}")

    return config | overrides | {"seed": seed, "device": device}
