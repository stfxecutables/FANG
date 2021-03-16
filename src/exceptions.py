class ConvolutionError(RuntimeError):
    pass


class VanishingError(ConvolutionError):
    pass


class ShapingError(RuntimeError):
    def __init__(self, message: str) -> None:
        message = f"{self.__class__.__name__}: {message}"
        super().__init__(message)
