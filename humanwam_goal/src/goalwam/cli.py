"""Command-line entrypoint for local files and HTTP serving."""

import argparse
import time


def main():
    parser = argparse.ArgumentParser(description="GoalWAM action inference")
    parser.add_argument("command", choices=("infer", "serve"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backend", choices=("native", "phyai"), default="native")
    parser.add_argument("--tp", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--sample")
    parser.add_argument("--output", default="outputs/actions.npz")
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.command == "infer" and not args.sample:
        parser.error("infer requires --sample")
    from .engine import GoalWAM

    with GoalWAM(
        args.checkpoint, backend=args.backend, tp=args.tp, cuda_graph=not args.no_cuda_graph
    ) as model:
        if args.command == "serve":
            from .server import serve

            serve(model, args.host, args.port)
        else:
            from .io import read_observation, save_actions

            source = read_observation(args.sample)
            started = time.perf_counter()
            result = model.predict(source)
            save_actions(args.output, result)
            print(
                f"Saved {result['normalized_actions'].shape[1]} actions to {args.output} ({time.perf_counter() - started:.3f} s, including first-request setup)"
            )


if __name__ == "__main__":
    main()
