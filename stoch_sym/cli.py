import argparse
import ast
import re
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # fmt: off
    parser.add_argument( "--dataset", choices=["cov", "linreg", "expm", "inv"])
    parser.add_argument("--group", type=str,
                        choices=["orth", "orth2"],
                        help="The group to use. Not all choices work with all datasets (see Section 7.3 of the paper).")
    parser.add_argument("--input-action", type=str,
                        choices=["lmul", "lmul-triv", "conj", "lmul-rmul"],
                        help="The action equipping the input space. Not all choices work with all datasets (see Section 7.3 of the paper).")
    parser.add_argument("--output-action", type=str,
                        choices=["conj", "lmul", "rmul-lmul", "rmul"],
                        help="The action equipping the output space. Not all choices work with all datasets (see Section 7.3 of the paper).")
    parser.add_argument("--backbone", choices=["mlp", "emlp"], help="The type of neural network to use for the backbone of the model")
    parser.add_argument("--gamma", choices=["none", "haar", "mlp-haar", "emlp"], help="The type of gamma to be used")

    parser.add_argument("--seed", default=int(time.time()), type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--config", default=[], action="append",
                        help="Config entries to override. "
                        "Specify as `key=value' (e.g. `--config learning_rate=1e-3'). "
                        "Multiple entries can be overriden by using this flag more than once. ")

    parser.add_argument("--resume", type=str) # TODO: Document

    parser.add_argument("--test", action="store_true") # TODO: Document
    parser.add_argument("--print-config", action="store_true", help="Print the config (disables training)")
    parser.add_argument("--draw-coarse-string-diagram", action="store_true", help='Draw the "coarse" string diagram of the model (disables training)')
    parser.add_argument("--draw-fine-string-diagram", action="store_true", help='Draw the "fine" string diagram of the model (disables training)')
    parser.add_argument("--print-model", action="store_true", help="Print the model (disables training)")
    parser.add_argument("--print-num-params", action="store_true", help="Print the number of parameters (disables training)")

    parser.add_argument("--wandb-project", default="stoch-sym", help="Weights & Biases project name")
    # fmt: on

    args = parser.parse_args()

    if not args.resume and not (
        args.dataset
        and args.group
        and args.input_action
        and args.output_action
        and args.backbone
        and args.gamma
    ):
        parser.error(
            "Must specify --resume, or all of --dataset, --group, --input-action, --output-action, --backbone, --gamma"
        )

    if args.resume and (
        args.dataset
        or args.group
        or args.input_action
        or args.output_action
        or args.backbone
        or args.gamma
    ):
        parser.error(
            "Cannot specify both --resume and any of --dataset, --group, --input-action, --output-action, --backbone, --gamma."
        )

    args.config_overrides = dict(
        parse_config_arg(parser, key_value) for key_value in args.config
    )

    return args


def parse_config_arg(parser: argparse.ArgumentParser, key_value: str):
    """Converts --config arguments into a key-value pair"""

    pattern = r"^\w+=.*$"

    if not re.match(pattern, key_value):
        parser.error(f"Invalid config item: {key_value}")

    k, v = key_value.split("=", maxsplit=1)

    try:
        v = ast.literal_eval(v)
    except:
        parser.error(f"Invalid config item: {key_value}")

    return k, v
