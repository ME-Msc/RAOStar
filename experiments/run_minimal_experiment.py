#!/usr/bin/env python3

"""Run a small, configurable R2D2 CC-POMDP experiment."""

import argparse

from models.r2d2model import R2D2Model
from raostar import RAOStar


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=1)
    parser.add_argument("--cc", type=float, default=0.08)
    parser.add_argument(
        "--observation-model",
        choices=["perfect", "noisy_position"],
        default="noisy_position",
    )
    parser.add_argument("--observation-accuracy", type=float, default=0.8)
    parser.add_argument("--move-success-probability", type=float, default=0.8)
    parser.add_argument("--fixed-horizon", type=int, default=4)
    parser.add_argument("--allow-left", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--show-policy", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    model = R2D2Model(
        length=args.length,
        DetermObs=args.observation_model == "perfect",
        obsProb=args.observation_accuracy,
        icy_move_forward_prob=args.move_success_probability,
        allow_left=args.allow_left,
    )
    algo = RAOStar(
        model,
        cc=args.cc,
        debugging=args.debug,
        cc_type="overall",
        fixed_horizon=args.fixed_horizon,
    )
    policy, graph = algo.search(model.initial_belief())

    mixed_belief_nodes = sum(
        len(node.state.belief) > 1 for node in graph.nodes.values()
    )
    root_action = graph.root.best_action

    print("INPUT:")
    print(f"  length = {args.length}")
    print(f"  cc = {args.cc}")
    print(f"  observation_model = {args.observation_model}")
    print(f"  observation_accuracy = {args.observation_accuracy}")
    print(f"  move_success_probability = {args.move_success_probability}")
    print(f"  fixed_horizon = {args.fixed_horizon}")
    print(f"  b_init = {model.initial_belief()}")

    print("OUTPUT SUMMARY:")
    print(f"  graph_nodes = {len(graph.nodes)}")
    print(f"  mixed_belief_nodes = {mixed_belief_nodes}")
    print(f"  policy_nodes = {len(policy)}")
    print(f"  root_value = {graph.root.value}")
    print(f"  root_exec_risk = {graph.root.exec_risk}")
    print(f"  root_action = {root_action.name if root_action else None}")

    if args.show_policy:
        print("POLICY:")
        for key, action in sorted(
            policy.items(), key=lambda item: (item[0][2], item[0][0])
        ):
            belief, probability, depth = key
            print(
                f"  depth={depth}, observation_probability={probability:.3f}, "
                f"belief={belief} -> action={action}"
            )


if __name__ == "__main__":
    main()
