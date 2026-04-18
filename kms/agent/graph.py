"""Compatibility shim for `kms.agent.graph`."""

import sys

from kms_agent.agent import graph as _graph

sys.modules[__name__] = _graph
