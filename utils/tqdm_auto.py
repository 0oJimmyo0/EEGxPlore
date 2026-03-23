"""tqdm wrapper: off by default when stderr is not a TTY (cluster batch logs)."""

from __future__ import annotations

import sys
from typing import Any, Iterable, Optional

from tqdm import tqdm as _tqdm


def tqdm_auto(iterable: Iterable, params: Optional[Any] = None, **kwargs: Any):
    """
    If params.use_tqdm is True -> always show.
    If params.use_tqdm is False -> never show.
    If params.use_tqdm is None (default) -> show only when stderr is a TTY.
    """
    disable = False
    if params is not None:
        u = getattr(params, "use_tqdm", None)
        if u is False:
            disable = True
        elif u is True:
            disable = False
        else:
            disable = not sys.stderr.isatty()
    else:
        disable = not sys.stderr.isatty()
    return _tqdm(iterable, disable=disable, **kwargs)
