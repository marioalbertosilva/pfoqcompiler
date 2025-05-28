"""
Module handling different error classes.
"""


class WidthError(Exception):
    """Raised when compiling a program with intractable call graph.

    Parameters
    ----------
    err_message: str
        Error message explaining and localizing the error.

    """
    def __init__(self, err_message: str, *args: object) -> None:
        super().__init__(*args)


class AncillaIndexError(Exception):
    """Raised when compiling a program with unsufficient ancilla qubits.

    Parameters
    ----------
    err_message: str
        Error message explaining and localizing the error.

    """
    def __init__(self, err_message: str, *args: object) -> None:
        super().__init__(*args)


class NotCompiledError(Exception):
    """Raised when checking a circuit that has not yet been compiled.

    Parameters
    ----------
    err_message: str
        Error message explaining and localizing the error.

    """
    def __init__(self, err_message: str, *args: object) -> None:
        super().__init__(*args)