"""Compatibility shim for `kms.agent.history`."""

import sys

from kms_agent.agent import history as _history

sys.modules[__name__] = _history
