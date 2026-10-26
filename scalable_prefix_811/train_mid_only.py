"""From-scratch Mid capacity: nine Grids and a 192-wide Decoder by default."""

from .single_training import build_parser, train


if __name__ == "__main__":
    train(build_parser("mid").parse_args())
