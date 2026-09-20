"""Make the test suite import this working tree, whatever pip has installed.

An editable install elsewhere on the machine puts its own ``src`` on
``sys.path`` at interpreter startup, so ``import onevoice`` can quietly
resolve to a different checkout. Tests then exercise code nobody is
editing: failures appear for APIs that exist here and passes hide
breakage that is real. Putting this tree first, then checking that the
import actually landed here, turns that silent mix-up into one loud
error at collection time.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"


def _put_this_tree_first() -> None:
    for path in (REPO_ROOT, SRC_ROOT):
        entry = str(path)
        while entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)


def _refuse_a_foreign_onevoice() -> None:
    import onevoice

    package = Path(onevoice.__file__).resolve()
    if SRC_ROOT not in package.parents:
        raise RuntimeError(
            f"'onevoice' was imported from {package.parent}, which is outside "
            f"this repository ({SRC_ROOT}).\n"
            "Something earlier on sys.path shadows this working tree -- most "
            "likely an editable install of another checkout. Reinstall from "
            "here with `pip install -e .`, or drop the stale .pth file from "
            "your environment's site-packages."
        )


_put_this_tree_first()
_refuse_a_foreign_onevoice()
