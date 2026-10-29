"""Strict wire models for tool-result safety classifier requests and decisions."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator

ClassifierCategory = Literal[
    "none",
    "instruction_override",
    "role_change",
    "prompt_extraction",
    "data_exfiltration",
    "tool_manipulation",
    "unknown",
]


class ClassifierRequest(BaseModel):
    """The only data sent to a safety classifier for one result chunk."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    tool_name: StrictStr = Field(alias="toolName")
    result_type: Literal["result", "error"] = Field(alias="resultType")
    chunk_index: StrictInt = Field(alias="chunkIndex", ge=0)
    chunk_count: StrictInt = Field(alias="chunkCount", gt=0)
    content: StrictStr

    @model_validator(mode="after")
    def validate_chunk_position(self) -> ClassifierRequest:
        if self.chunk_index >= self.chunk_count:
            raise ValueError("chunkIndex must be less than chunkCount")
        return self


class ClassifierDecision(BaseModel):
    """Strict classifier output with no free-form explanation."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    injection_detected: StrictBool = Field(alias="injectionDetected")
    category: ClassifierCategory

    @model_validator(mode="after")
    def validate_category_consistency(self) -> ClassifierDecision:
        if not self.injection_detected and self.category != "none":
            raise ValueError("benign decisions must use category 'none'")
        if self.injection_detected and self.category == "none":
            raise ValueError("malicious decisions must use a non-none category")
        return self
