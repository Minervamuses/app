"""Compatibility shim for `kms.agent.state`."""

import sys

from kms_agent.agent import state as _state

sys.modules[__name__] = _state
