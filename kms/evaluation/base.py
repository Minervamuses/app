"""Compatibility shim for `kms.evaluation.base`."""

import sys

from kms_agent.evaluation import base as _base

sys.modules[__name__] = _base
