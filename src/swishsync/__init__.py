"""SwishSync platform (Phase A refactor).

Model-agnostic, IR-mediated basketball CV pipeline. Coexists with the legacy
`swishsync_cv` package during migration; the shot engine is reused via an
adapter (see swishsync.events.shot).
"""

from swishsync.core import schemas  # noqa: F401
# import side-effect: register built-in plugins
from swishsync.events import shot  # noqa: F401,E402

__all__ = ["schemas"]
