"""Patch argbind.parse_args to accept an explicit `argv` argument.

Original signature:
    def parse_args(p=None, group="default")
    -> always reads sys.argv

After patch:
    def parse_args(p=None, group="default", argv=None)
    -> if argv is None, falls back to sys.argv (unchanged behaviour)
    -> if argv is provided (e.g. sys.argv, or a hand-crafted subset),
       uses that list to compute used_args AND drives argparse with argv[1:],
       matching the convention that argv[0] is the program name.

Idempotent: re-running the patch is a no-op (asserts on the unpatched marker).

The argbind module location is discovered dynamically, so this script works
across base images (miniforge in the Tencent image, conda env in the public
PyTorch image, system Python, virtualenvs, etc).
"""

from pathlib import Path
import sys


def locate_target() -> Path:
    try:
        import argbind
    except ImportError:
        sys.exit("argbind is not importable in this Python environment")
    # argbind.__file__ points at the package __init__.py, but parse_args is
    # defined in argbind/argbind.py — use inspect to find the function's actual
    # source file regardless of the package layout.
    import inspect
    src = inspect.getsourcefile(argbind.parse_args)
    if src is None:
        sys.exit("could not locate argbind.parse_args source file")
    target = Path(src)
    if not target.is_file():
        sys.exit(f"parse_args source file {target} is not on disk")
    return target


TARGET = locate_target()

OLD_SIG = 'def parse_args(p=None, group: Union[list, str] = "default"):'
NEW_SIG = 'def parse_args(p=None, group: Union[list, str] = "default", argv: list = None):'

OLD_BLOCK = """    p = build_parser(group=group) if p is None else p
    used_args = [x.replace('--', '').split('=')[0] for x in sys.argv if x.startswith('--')]
    used_args.extend(['args.save', 'args.load'])

    known, unknown = p.parse_known_args()"""

NEW_BLOCK = """    p = build_parser(group=group) if p is None else p
    # patched: allow caller to pass an explicit argv (sys.argv or a subset).
    # When argv is None we fall back to sys.argv, preserving original behaviour.
    if argv is None:
        argv = sys.argv
    used_args = [x.replace('--', '').split('=')[0] for x in argv if x.startswith('--')]
    used_args.extend(['args.save', 'args.load'])

    # argparse's parse_known_args() defaults to sys.argv[1:]; mirror that
    # convention by stripping the leading program-name token from argv.
    known, unknown = p.parse_known_args(args=argv[1:])"""


def main() -> None:
    text = TARGET.read_text()

    if NEW_SIG in text:
        print(f"[skip] {TARGET} already patched")
        return

    if OLD_SIG not in text:
        sys.exit(f"signature not found in {TARGET}")
    if OLD_BLOCK not in text:
        sys.exit(f"expected body block not found in {TARGET}")

    text = text.replace(OLD_SIG, NEW_SIG, 1)
    text = text.replace(OLD_BLOCK, NEW_BLOCK, 1)
    TARGET.write_text(text)
    print(f"[ok] patched {TARGET}")


if __name__ == "__main__":
    main()
