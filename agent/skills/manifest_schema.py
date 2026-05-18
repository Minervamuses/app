"""Schema validation for skill manifests."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError, model_validator


class OptionalCapability(BaseModel):
    """Structured optional capability declaration."""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: str
    use_when: str | None = None


class Capabilities(BaseModel):
    """Capability requirements declared by a skill."""

    model_config = ConfigDict(extra="forbid", strict=True)

    required: list[str] = Field(default_factory=list)
    optional: list[str | OptionalCapability] = Field(default_factory=list)


class SkillResource(BaseModel):
    """Resource declaration in a skill manifest."""

    model_config = ConfigDict(extra="forbid", strict=True)

    path: str
    use_when: str | None = None
    pinned: StrictBool = False


class ToolPolicy(BaseModel):
    """Tool policy declaration in a skill manifest."""

    model_config = ConfigDict(extra="forbid", strict=True)

    disallow: list[str] = Field(default_factory=list)


class SkillManifest(BaseModel):
    """Validated skill manifest schema."""

    model_config = ConfigDict(extra="forbid", strict=True)

    capabilities: Capabilities | None = None
    resources: list[SkillResource] = Field(default_factory=list)
    task_modes: list[str] = Field(default_factory=list)
    tool_policy: ToolPolicy | None = None

    @model_validator(mode="after")
    def reject_empty_capabilities_without_policy(self) -> "SkillManifest":
        has_empty_capabilities = (
            self.capabilities is not None
            and not self.capabilities.required
            and not self.capabilities.optional
        )
        has_disallow_policy = bool(self.tool_policy and self.tool_policy.disallow)
        if has_empty_capabilities and not has_disallow_policy:
            raise ValueError(
                "capabilities must not be empty unless tool_policy.disallow is set"
            )
        return self


def validate_skill_manifest(
    data: Mapping[str, Any],
    *,
    source: str | Path = "manifest.yaml",
) -> dict[str, Any]:
    """Validate a manifest mapping and return a normalized plain dict."""
    try:
        manifest = SkillManifest.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"invalid skill manifest: {source}: {exc}") from exc
    return manifest.model_dump(mode="python", exclude_none=True)
