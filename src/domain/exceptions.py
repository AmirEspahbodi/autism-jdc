"""Domain-level exceptions for the JDC project.

All custom exceptions inherit from JDCException to allow unified error
handling at application boundaries.
"""


class JDCException(Exception):
    """Base exception for all JDC project errors.

    Args:
        message: Human-readable error description.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return f"[{self.__class__.__name__}] {self.message}"


class CUDANotAvailableError(JDCException):
    """Raised when CUDA is not available on the host machine.

    Args:
        message: Details about the GPU/CUDA availability failure.
    """


class DatasetLoadError(JDCException):
    """Raised when a dataset file cannot be opened or parsed.

    Args:
        message: Path and reason for the failure.
    """


class MalformedOutputError(JDCException):
    """Raised when a model_output string cannot be parsed as valid JSON.

    Args:
        message: Sample id and raw string that caused the failure.
    """


class ModelLoadError(JDCException):
    """Raised when the language model or tokenizer fails to load.

    Args:
        message: Model name/path and underlying error.
    """


class InferenceError(JDCException):
    """Raised when model inference or output parsing fails.

    Args:
        message: Description of the failure.
        raw_output: The raw generated text that could not be parsed.
    """

    def __init__(self, message: str, raw_output: str = "") -> None:
        super().__init__(message)
        self.raw_output = raw_output


class ConfigurationError(JDCException):
    """Raised when the configuration file is missing or contains invalid values.

    Args:
        message: Which field/file is invalid and why.
    """
