"""Compatibility shim for `kms.agent.session`."""

import sys

from kms_agent.agent import session as _session

sys.modules[__name__] = _session
