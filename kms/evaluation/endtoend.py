"""Compatibility shim for `kms.evaluation.endtoend`."""

import sys

from kms_agent.evaluation import endtoend as _endtoend

sys.modules[__name__] = _endtoend
