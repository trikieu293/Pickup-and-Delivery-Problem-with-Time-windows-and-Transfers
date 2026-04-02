"""Custom exceptions for PDPTWT solver."""


class PDPTWTException(Exception):
    """Base exception for all PDPTWT-related errors."""
    pass


class DataValidationError(PDPTWTException):
    """Raised when data validation fails."""
    pass


class DataLoadError(PDPTWTException):
    """Raised when data loading fails."""
    pass


class ModelBuildError(PDPTWTException):
    """Raised when model building fails."""
    pass


class SolverError(PDPTWTException):
    """Base exception for solver-related errors."""
    pass


class ModelInfeasibleError(SolverError):
    """Raised when the model is infeasible."""
    pass


class ModelUnboundedError(SolverError):
    """Raised when the model is unbounded."""
    pass


class SolverTimeoutError(SolverError):
    """Raised when solver reaches time limit without finding optimal solution."""
    pass


class SolverConfigError(SolverError):
    """Raised when solver configuration is invalid."""
    pass


class SolutionExtractionError(PDPTWTException):
    """Raised when solution extraction fails."""
    pass


class VisualizationError(PDPTWTException):
    """Raised when visualization fails."""
    pass
