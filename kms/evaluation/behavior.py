"""Compatibility shim for `kms.evaluation.behavior`."""

import sys

from kms_agent.evaluation import behavior as _behavior

sys.modules[__name__] = _behavior
