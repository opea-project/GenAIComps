from .accuracy import cli_evaluate as evaluate
from .arguments import LMEvalParser, setup_parser

__all__ = [evaluate, LMEvalParser, setup_parser]