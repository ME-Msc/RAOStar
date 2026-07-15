#!/usr/bin/env python3

"""Run RAO* on R2D2 and optionally visualize the resulting policy."""

import argparse
from pathlib import Path

from models.r2d2model import R2D2Model
from raostar import RAOStar

HTML_OUTPUT = Path("results/r2d2_policy.html")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc", type=float, default=0.08)
    parser.add_argument("--length", type=int, default=3)
    parser.add_argument(
        "--observation-model",
        choices=["perfect", "noisy_position"],
        default="perfect",
    )
    parser.add_argument("--observation-accuracy", type=float, default=0.8)
    parser.add_argument("--move-success-probability", type=float, default=0.8)
    parser.add_argument("--fixed-horizon", type=int)
    parser.add_argument("--allow-left", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    return parser.parse_args()


def create_planner(model, args):
    planner_options = {
        "cc": args.cc,
        "debugging": args.debug,
        "cc_type": "overall",
    }
    if args.fixed_horizon is not None:
        planner_options["fixed_horizon"] = args.fixed_horizon
    return RAOStar(model, **planner_options)


def visualize_policy(model, graph):
    from simulators.r2d2_visualizer import R2D2Visualizer

    visualizer = R2D2Visualizer(model)
    output_path = visualizer.write_html(graph, HTML_OUTPUT)
    print(f"interactive policy: {output_path}")


def main():
    args = parse_args()
    model = R2D2Model(
        length=args.length,
        DetermObs=args.observation_model == "perfect",
        obsProb=args.observation_accuracy,
        icy_move_forward_prob=args.move_success_probability,
        allow_left=args.allow_left,
    )
    planner = create_planner(model, args)
    policy, graph = planner.search(model.initial_belief())

    model.print_model()
    model.print_policy(policy)
    print("\n** Result summary **")
    print(f"chance constraint: {args.cc}")
    print(f"policy nodes: {len(policy)}")
    print(f"graph nodes: {len(graph.nodes)}")
    print(f"root value: {graph.root.value}")
    print(f"root execution risk: {graph.root.exec_risk}")

    if args.visualize:
        visualize_policy(model, graph)


if __name__ == "__main__":
    main()
