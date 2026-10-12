"""Lazy registration in the API process and spawned vLLM workers."""

import logging

ARCHITECTURE = "JevServeQwen35ForSequenceClassification"


def register() -> None:
    from vllm.model_executor.models import ModelRegistry

    # A failed scheduler must not dump token IDs or request identifiers.
    logging.getLogger("vllm.logging_utils.dump_input").disabled = True

    ModelRegistry.register_model(
        ARCHITECTURE, "jevserve.runtime.model:JevServeQwen35ForSequenceClassification"
    )
