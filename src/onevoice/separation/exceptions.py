

from __future__ import annotations

class BackendError(Exception):
    pass

class BackendNotFoundError(BackendError):
    pass

class BackendConfigError(BackendError):
    pass

class BackendInitializationError(BackendError):
    pass

class BackendOOMError(BackendError):
    pass

class InferenceTimeoutError(BackendError):
    pass
