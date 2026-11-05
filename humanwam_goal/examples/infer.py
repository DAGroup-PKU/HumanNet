"""Load an observation, run the GoalWAM Python API, and save the predicted actions."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--backend", choices=("native", "phyai"), default="native")
    parser.add_argument("--tp", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--output", default="outputs/actions.npz")
    parser.add_argument("--no-cuda-graph", action="store_true")
    args = parser.parse_args()

    from goalwam import GoalWAM
    from goalwam.io import read_observation, save_actions

    observation = read_observation(args.sample)
    with GoalWAM(
        args.checkpoint, backend=args.backend, tp=args.tp, cuda_graph=not args.no_cuda_graph
    ) as model:
        result = model.predict(observation)
    save_actions(args.output, result)
    print(f"Saved {result['normalized_actions'].shape[1]} actions to {args.output}")


if __name__ == "__main__":
    main()
