import logging
logger = logging.getLogger("DynamicCodec")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] | [%(levelname)s] | %(name)s | %(message)s"
)


def log_components(args):
    """Pretty-print the dynamic component selection (state + 5 module choices).

    One logger.info call with embedded newlines, so the timestamp prefix only
    appears on the first line and the block reads as a single banner.
    """
    fields = [
        ("state",        args.get("state")),
        ("input_format", args.get("input_format")),
        ("encoder",      args.get("encoder")),
        ("quantizer",    args.get("quantizer")),
        ("decoder",      args.get("decoder")),
        ("vocoder",      args.get("vocoder")),
    ]
    width = max(len(k) for k, _ in fields)
    body = "\n".join(f"  {k:<{width}} : {v}" for k, v in fields)
    bar = "=" * 40
    logger.info("DynamicCodec components\n%s\n%s\n%s", bar, body, bar)
