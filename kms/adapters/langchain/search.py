"""Compatibility shim for `kms.adapters.langchain.search`."""

import sys

from kms_agent.adapters.langchain import search as _search

sys.modules[__name__] = _search
