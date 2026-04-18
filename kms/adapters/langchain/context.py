"""Compatibility shim for `kms.adapters.langchain.context`."""

import sys

from kms_agent.adapters.langchain import context as _context

sys.modules[__name__] = _context
