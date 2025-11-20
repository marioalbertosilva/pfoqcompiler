"""
Module handling different error classes.
"""


class WellFoundedError(Exception):
    """Raised when there is a cycle in the call graph that does not reduce the qubit input size.

    Parameters
    ----------
    err_message: str
        Error message explaining and localizing the error.

    """
    def __init__(self, err_message: str, *args: object) -> None:
        super().__init__(*args)


class WidthError(Exception):
    """Raised when a procedure performs more than one recursive call per branch.

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