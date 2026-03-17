import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    # Use the specified level (default INFO) for the root logger,
    # but set app.services loggers to DEBUG for full pipeline tracing.
    effective_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=effective_level,
        format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )

    # Force all app.services loggers to DEBUG for pipeline tracing
    logging.getLogger("app.services").setLevel(logging.DEBUG)
