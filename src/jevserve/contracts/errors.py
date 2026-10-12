"""Public error codes. Messages must never contain raw input or exception text."""

from enum import StrEnum

from jevserve.contracts.requests import Contract, Text


class ErrorCode(StrEnum):
    INVALID_REQUEST = "invalid_request"
    BODY_TOO_LARGE = "body_too_large"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    CONTEXT_TOO_LONG = "context_too_long"
    OVERLOADED = "overloaded"
    NOT_READY = "not_ready"
    ENGINE_FAILED = "engine_failed"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    UNAUTHORIZED = "unauthorized"
    INVALID_MODEL_OUTPUT = "invalid_model_output"


HTTP_STATUS = {
    ErrorCode.INVALID_REQUEST: 400,
    ErrorCode.BODY_TOO_LARGE: 413,
    ErrorCode.UNSUPPORTED_CAPABILITY: 422,
    ErrorCode.CONTEXT_TOO_LONG: 422,
    ErrorCode.OVERLOADED: 429,
    ErrorCode.NOT_READY: 503,
    ErrorCode.ENGINE_FAILED: 503,
    ErrorCode.DEADLINE_EXCEEDED: 504,
    ErrorCode.UNAUTHORIZED: 401,
    ErrorCode.INVALID_MODEL_OUTPUT: 503,
}


class ErrorDetail(Contract):
    code: ErrorCode
    message: Text


class ErrorResponse(Contract):
    error: ErrorDetail
    request_id: Text
