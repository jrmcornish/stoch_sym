#!/usr/bin/env python


def main():
    from stoch_sym.cli import parse_args
    from stoch_sym.config import get_config, apply_config_overrides

    args = parse_args()

    if args.resume:
        import wandb

        config: dict = wandb.Api().run(f"{args.wandb_project}/{args.resume}").config

    else:
        config = get_config(
            dataset=args.dataset,
            backbone=args.backbone,
            group=args.group,
            input_action=args.input_action,
            output_action=args.output_action,
            gamma=args.gamma,
        )

    config = apply_config_overrides(
        config=config,
        overrides=args.config_overrides,
        seed=args.seed,
        device=args.device,
    )

    if args.print_config:
        print_config(config)

    elif args.draw_coarse_string_diagram:
        draw_coarse_string_diagram(config)

    elif args.draw_fine_string_diagram:
        draw_fine_string_diagram(config)

    elif args.print_num_params:
        print_num_params(config)

    elif args.print_model:
        print_model(config)

    elif args.test:
        test(args, config)

    else:
        train(args, config)


def print_config(config):
    import json

    print(json.dumps(config, indent=4))


def draw_coarse_string_diagram(config):
    from stoch_sym.model.builder import get_coarse_string_diagram

    get_coarse_string_diagram(config["gamma"]).dagger().draw(asymmetry=0)


def draw_fine_string_diagram(config):
    from stoch_sym.model.builder import get_fine_string_diagram

    get_fine_string_diagram(config).dagger().draw(asymmetry=0)


def print_num_params(config):
    from stoch_sym.experiment import setup_experiment

    exp = setup_experiment(config=config)

    print(f"Number of parameters: {get_num_params(exp.model)}")


def print_model(config):
    from stoch_sym.experiment import setup_experiment

    exp = setup_experiment(config=config)

    print(exp.model)


def train(args, config):
    import contextlib

    import wandb

    from stoch_sym.experiment import setup_experiment
    from stoch_sym.train import train

    exp = setup_experiment(config=config)

    if args.resume:
        start_epoch = wandb_resume(exp, args.wandb_project, args.resume, testing=False)

    else:
        wandb.init(
            project=args.wandb_project,
            name=f"{config['dataset']}_{config['group']}_{config['input_action']}_{config['output_action']}_{config['backbone']}_{config['gamma']}",
            config=config | {"num_params": get_num_params(exp.model)},
        )

        start_epoch = 0

        print(f"Number of parameters: {get_num_params(exp.model)}")

    print(f"Device: {exp.device}")
    print(f"Seed: {args.seed}")

    with contextlib.suppress(KeyboardInterrupt):
        train(
            model=exp.model,
            train_loader=exp.train_loader,
            test_loader=exp.test_loader,
            optimiser=exp.optimiser,
            loss_fn=exp.loss_fn,
            test_metrics=exp.test_metrics,
            start_epoch=start_epoch,
            num_epochs=config["num_epochs"],
            epochs_per_test=config["epochs_per_test"],
            device=exp.device,
        )


def test(args, config):
    import json

    from stoch_sym.experiment import setup_experiment
    from stoch_sym.train import test

    exp = setup_experiment(config=config)

    if args.resume:
        wandb_resume(exp, args.wandb_project, args.resume, testing=True)

    print(f"Device: {exp.device}")
    print(f"Seed: {args.seed}")

    test_metrics = test(
        model=exp.model,
        test_loader=exp.test_loader,
        test_metrics=exp.test_metrics,
        device=exp.device,
    )

    print(json.dumps(test_metrics))


def wandb_resume(exp, wandb_project: str, wandb_id: str, testing: bool):
    import wandb

    from stoch_sym.checkpointing import load_checkpoint

    run = wandb.Api().run(path=f"{wandb_project}/{wandb_id}")

    # NOTE: This check is not perfect, since there is a reasonable delay between
    # when a run is resumed and when its state changes to "running". But it is
    # better than nothing. In the case where the same run is resumed multiple
    # times, it appears that the later values that are logged overwrite the
    # earlier ones.
    if run.state not in ["finished", "crashed", "killed", "failed"] and not testing:
        raise IOError(f"Cannot resume run while its state is `{run.state}'")

    wandb.init(
        project=wandb_project,
        id=wandb_id,
        resume="must",
        mode="disabled" if testing else None,
    )

    last_epoch = load_checkpoint(run=run, model=exp.model, optimiser=exp.optimiser)

    print(f"Resuming after epoch {last_epoch}")

    start_epoch = last_epoch + 1

    return start_epoch


def get_num_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    main()
