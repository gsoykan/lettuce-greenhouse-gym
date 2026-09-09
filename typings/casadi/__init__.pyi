# CasADi 3.8 ships a stub file mypy cannot parse (a duplicate parameter name), which aborts every
# run. This minimal stub takes precedence through `mypy_path` and types the module as dynamic, which
# is how it behaves anyway: symbolic expressions accept whatever arithmetic is asked of them.
from typing import Any

def __getattr__(name: str) -> Any: ...
