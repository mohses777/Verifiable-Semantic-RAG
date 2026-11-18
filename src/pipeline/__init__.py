"""End-to-end document processing pipeline."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.pipeline.runner import (
        DocumentProcessingPipeline,
        DocumentSummary,
        PipelineArtifacts,
        PipelineSummary,
        estimate_processing_cost,
    )


def __getattr__(name: str):
    if name in {
        "DocumentProcessingPipeline",
        "DocumentSummary",
        "PipelineArtifacts",
        "PipelineSummary",
        "estimate_processing_cost",
    }:
        import importlib
        runner = importlib.import_module("src.pipeline.runner")
        return getattr(runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DocumentProcessingPipeline",
    "DocumentSummary",
    "PipelineArtifacts",
    "PipelineSummary",
    "estimate_processing_cost",
]
