"""From-scratch Base capacity: eight Grids and a 128-wide Decoder by default."""

from .single_training import build_parser, train


if __name__ == "__main__":
    train(build_parser("base").parse_args())
