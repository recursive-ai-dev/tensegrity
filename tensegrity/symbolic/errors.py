"""Error types for the symbolic package."""


class SymbolicError(Exception):
    """Base class for all symbolic-package errors."""


class _CodedError(SymbolicError):
    kind = ""

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}" if detail else code)


class ParseError(_CodedError):
    kind = "parse"


class TypeError_(_CodedError):
    kind = "type"


class EvalError(_CodedError):
    kind = "eval"


class ConfigError(SymbolicError):
    pass


class CorpusError(SymbolicError):
    pass
