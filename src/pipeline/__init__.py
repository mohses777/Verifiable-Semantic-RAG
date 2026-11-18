"""End-to-end document processing and question-answering pipelines."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.pipeline.qa import QAPipelineSummary, QuestionAnsweringPipeline
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
    if name in {"QAPipelineSummary", "QuestionAnsweringPipeline"}:
        import importlib
        qa = importlib.import_module("src.pipeline.qa")
        return getattr(qa, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DocumentProcessingPipeline",
    "DocumentSummary",
    "PipelineArtifacts",
    "PipelineSummary",
    "QAPipelineSummary",
    "QuestionAnsweringPipeline",
    "estimate_processing_cost",
]
