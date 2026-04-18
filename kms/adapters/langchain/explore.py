"""Compatibility shim for `kms.adapters.langchain.explore`."""

import sys

from kms_agent.adapters.langchain import explore as _explore

sys.modules[__name__] = _explore
