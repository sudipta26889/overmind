class MaskingError(RuntimeError):
    """Masking could not be completed, so nothing may be sent to a remote model.

    ``reason`` is a short code (counted per day); it never contains the text that failed.
    """

    def __init__(self, reason: str, *, leak: bool = False):
        super().__init__(reason)
        self.reason = reason
        # A leak is a gateway guardrail refusal: something sensitive got as far as the request.
        self.leak = leak
