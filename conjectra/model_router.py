"""Capability-verified model & effort router and NL policy compiler for conjectra."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from conjectra.types import (
    ConstraintStrength,
    HypothesisNode,
    ModelPolicy,
    ResearchGoal,
    ResolvedModelRoute,
    ReviewTier,
    ScientificRole,
)

if TYPE_CHECKING:
    from conjectra.arbiter import ResourceArbiter
    from conjectra.state_db import StateDB

DEFAULT_CAPABILITIES_PATH = Path(
    os.environ.get("CONJECTRA_PROVIDERS") or (Path(__file__).resolve().parent / "data" / "providers.json")
)

VALID_EFFORT_TOKENS: frozenset[str] = frozenset(
    {"off", "minimal", "low", "medium", "high", "xhigh", "max", "extra high", "default", "custom"}
)

UNKNOWN_EFFORT_TRAP_TOKENS: frozenset[str] = frozenset(
    {"ultra", "turbo", "extreme", "super", "hyper", "insane", "infinite", "auto_max"}
)

REVIEWER_ROLE_VALUES: frozenset[str] = frozenset(
    {
        ScientificRole.DESIGN_REVIEWER.value,
        ScientificRole.IMPL_REVIEWER.value,
        ScientificRole.RESULT_AUDITOR.value,
    }
)

ROLE_ALIASES: dict[str, str] = {
    "research_director": ScientificRole.RESEARCH_DIRECTOR.value,
    "director": ScientificRole.RESEARCH_DIRECTOR.value,
    "l2": ScientificRole.RESEARCH_DIRECTOR.value,
    "research_lead": ScientificRole.RESEARCH_LEAD.value,
    "lead": ScientificRole.RESEARCH_LEAD.value,
    "proposer": ScientificRole.RESEARCH_LEAD.value,
    "l1": ScientificRole.RESEARCH_LEAD.value,
    "theorist": ScientificRole.THEORIST.value,
    "experiment_designer": ScientificRole.EXPERIMENT_DESIGNER.value,
    "designer": ScientificRole.EXPERIMENT_DESIGNER.value,
    "engineer_builder": ScientificRole.ENGINEER_BUILDER.value,
    "builder": ScientificRole.ENGINEER_BUILDER.value,
    "worker": ScientificRole.ENGINEER_BUILDER.value,
    "prescreener": ScientificRole.IMPL_REVIEWER.value,
    "l1_batch": ScientificRole.IMPL_REVIEWER.value,
    "design_reviewer": ScientificRole.DESIGN_REVIEWER.value,
    "impl_reviewer": ScientificRole.IMPL_REVIEWER.value,
    "implementation_reviewer": ScientificRole.IMPL_REVIEWER.value,
    "adversary": ScientificRole.IMPL_REVIEWER.value,
    "verifier": ScientificRole.IMPL_REVIEWER.value,
    "result_auditor": ScientificRole.RESULT_AUDITOR.value,
    "blind_auditor": ScientificRole.RESULT_AUDITOR.value,
    "auditor": ScientificRole.RESULT_AUDITOR.value,
}

# Ordered longest-first for deterministic NL scanning.
_NL_MODEL_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"deepseek-chat", re.IGNORECASE), "deepseek-chat"),
    (re.compile(r"grok-code-fast", re.IGNORECASE), "grok-code-fast"),
    (re.compile(r"webchat-gpt-xhigh", re.IGNORECASE), "webchat-gpt-xhigh"),
    (re.compile(r"chatgpt\s+extra\s+high", re.IGNORECASE), "webchat-gpt-xhigh"),
    (re.compile(r"\bclaude-opus-5-5\b", re.IGNORECASE), "claude-opus-5-5"),
    (re.compile(r"\bclaude-sonnet-5-5\b", re.IGNORECASE), "claude-sonnet-5-5"),
    (re.compile(r"\bgpt-5-pro\b", re.IGNORECASE), "gpt-5-pro"),
    (re.compile(r"\bgpt-5-codex\b", re.IGNORECASE), "gpt-5-codex"),
    (re.compile(r"\bgemini-pro(?:-(?:low|medium|high|xhigh))?\b", re.IGNORECASE), "gemini-pro"),
    (re.compile(r"\bgemini-flash(?:-(?:low|medium|high|xhigh))?\b", re.IGNORECASE), "gemini-flash"),
    (re.compile(r"\bgemini-ultra(?:-(?:low|medium|high))?\b", re.IGNORECASE), "gemini-ultra"),
    (re.compile(r"\bdeepseek\b", re.IGNORECASE), "deepseek-chat"),
    (re.compile(r"\bgrok\b", re.IGNORECASE), "grok-code-fast"),
    (re.compile(r"\bclaude-opus\b", re.IGNORECASE), "claude-opus-5-5"),
    (re.compile(r"\bclaude-sonnet\b", re.IGNORECASE), "claude-sonnet-5-5"),
    (re.compile(r"\bopus\b", re.IGNORECASE), "claude-opus-5-5"),
    (re.compile(r"\bsonnet\b", re.IGNORECASE), "claude-sonnet-5-5"),
    (re.compile(r"\bwebchat\b", re.IGNORECASE), "webchat-gpt-xhigh"),
)

_TRAILING_EFFORT_RE = re.compile(
    r"^\s*[:/\-\s]?\s*(extra\s+high|xhigh|minimal|medium|high|low|max|off|ultra|turbo|extreme|super|hyper|insane)\b",
    re.IGNORECASE,
)

_GEMINI_SUFFIX_RE = re.compile(
    r"^(?P<base>gemini-(?:pro|flash|ultra))-(?P<eff>low|medium|high|xhigh|max)$",
    re.IGNORECASE,
)

_RAW_HQ_RESOURCE_NAMES: frozenset[str] = frozenset(
    {
        "llm_claude",
        "llmclaude",
        "llm_gemini",
        "llmgemini",
        "llm_codex_audit",
        "llmcodexaudit",
        "llm_deepseek",
        "llmdeepseek",
    }
)


class PolicyBlockedError(RuntimeError, ValueError):
    """Raised when a routing request violates a HARD policy, capability cap, or pool quota."""

    def __init__(
        self,
        reason: str,
        *,
        waitable: bool = False,
        rule_id: str = "",
        code: str = "BLOCKED_POLICY",
        pinned_route: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = str(reason)
        self.waitable = bool(waitable)
        self.rule_id = str(rule_id)
        self.code = str(code)
        self.pinned_route: dict[str, Any] | None = (
            dict(pinned_route) if isinstance(pinned_route, Mapping) else None
        )


_ALLOWED_POLICY_KEYS: frozenset[str] = frozenset(
    {
        "source_nl",
        "routing_policy_nl",
        "project",
        "default_strength",
        "allow_effort_alias",
        "foreground_reserve_quota_pct",
        "rules",
        "review_overrides",
        "forced",
        "hard",
        "allow_fallback",
        "disallowed",
        "disallowed_models",
        "disallowed_families",
        "disallowed_pools",
        "disallowed_bundle",
        "allowed_custom_commands",
        "global_policy",
        "direction_policy",
        "has_active_constraints",
    }
)

_ALLOWED_RULE_KEYS: frozenset[str] = frozenset(
    {
        "rule_id",
        "rank",
        "roles",
        "role",
        "tags",
        "tag",
        "candidates",
        "spec",
        "model",
        "provider",
        "effort",
        "strength",
        "forced",
        "hard",
        "allow_fallback",
        "allow_effort_alias",
        "author_only",
        "project_scope",
        "clause",
        "disallowed",
        "disallowed_models",
        "disallowed_families",
        "disallowed_pools",
        "disallowed_bundle",
        "_scope_level",
        "_scope_name",
    }
)

_ALLOWED_CANDIDATE_KEYS: frozenset[str] = frozenset(
    {"spec", "provider", "provider_key", "model", "canonical_model", "effort", "family"}
)

_KNOWN_FAMILY_TOKENS: dict[str, str] = {
    "anthropic": "anthropic",
    "claude": "anthropic",
    "openai": "openai",
    "codex": "openai",
    "gpt": "openai",
    "google": "google",
    "gemini": "google",
    "deepseek": "deepseek",
    "xai": "xai",
    "grok": "xai",
}

_KNOWN_POOL_TOKENS: dict[str, set[str]] = {
    "claude": {"claude", "llm_claude"},
    "llm_claude": {"claude", "llm_claude"},
    "llmclaude": {"claude", "llm_claude"},
    "codex": {"codex", "llm_codex_audit"},
    "llm_codex_audit": {"codex", "llm_codex_audit"},
    "llmcodexaudit": {"codex", "llm_codex_audit"},
    "gemini": {"gemini", "llm_gemini"},
    "gemini-cli": {"gemini", "llm_gemini"},
    "llm_gemini": {"gemini", "llm_gemini"},
    "llmgemini": {"gemini", "llm_gemini"},
    "deepseek": {"deepseek", "llm_deepseek"},
    "llm_deepseek": {"deepseek", "llm_deepseek"},
    "llmdeepseek": {"deepseek", "llm_deepseek"},
    "grok": {"grok"},
    "deepseek-local": {"deepseek-local", "deepseek_local"},
    "deepseek_local": {"deepseek-local", "deepseek_local"},
    "webchat": {"webchat", "web_chat_audit"},
    "web_chat_audit": {"webchat", "web_chat_audit"},
}


def _check_allowed_mapping_keys(
    mapping: Mapping[str, Any], allowed: frozenset[str], *, scope_label: str
) -> None:
    unknown = sorted(str(k) for k in mapping.keys() if str(k) not in allowed)
    if unknown:
        raise PolicyBlockedError(
            f"BLOCKED_INVALID_POLICY: unsupported routing constraint key(s) {unknown} in {scope_label}",
            waitable=False,
            rule_id=scope_label,
            code="BLOCKED_INVALID_POLICY",
        )


def _normalize_strength_from_mapping(
    mapping: Mapping[str, Any],
    *,
    fallback_strength: ConstraintStrength,
    scope_label: str,
    strength_key: str = "strength",
) -> ConstraintStrength:
    for bkey in ("forced", "hard", "allow_fallback"):
        if bkey in mapping and mapping[bkey] is not None and not isinstance(mapping[bkey], bool):
            raise PolicyBlockedError(
                f"BLOCKED_INVALID_POLICY: {bkey!r} in {scope_label} must be a boolean, got {mapping[bkey]!r}",
                waitable=False,
                rule_id=scope_label,
                code="BLOCKED_INVALID_POLICY",
            )

    forced_val = mapping.get("forced")
    hard_val = mapping.get("hard")
    allow_fb_val = mapping.get("allow_fallback")

    implies_hard = (forced_val is True) or (hard_val is True) or (allow_fb_val is False)
    if (forced_val is True or hard_val is True) and allow_fb_val is True:
        raise PolicyBlockedError(
            f"BLOCKED_INVALID_POLICY: contradictory constraints forced/hard=True with allow_fallback=True in {scope_label}",
            waitable=False,
            rule_id=scope_label,
            code="BLOCKED_INVALID_POLICY",
        )
    if (forced_val is False or hard_val is False) and allow_fb_val is False:
        raise PolicyBlockedError(
            f"BLOCKED_INVALID_POLICY: contradictory constraints forced/hard=False with allow_fallback=False in {scope_label}",
            waitable=False,
            rule_id=scope_label,
            code="BLOCKED_INVALID_POLICY",
        )

    explicit_raw = mapping.get(strength_key)
    if explicit_raw is not None:
        try:
            explicit_strength = (
                explicit_raw
                if isinstance(explicit_raw, ConstraintStrength)
                else ConstraintStrength(str(explicit_raw).strip().upper())
            )
        except ValueError as exc:
            raise PolicyBlockedError(
                f"BLOCKED_INVALID_POLICY: invalid {strength_key}={explicit_raw!r} in {scope_label}",
                waitable=False,
                rule_id=scope_label,
                code="BLOCKED_INVALID_POLICY",
            ) from exc
        if implies_hard and explicit_strength == ConstraintStrength.SOFT_FALLBACK:
            raise PolicyBlockedError(
                f"BLOCKED_INVALID_POLICY: contradictory forced/hard/allow_fallback=False with {strength_key}=SOFT_FALLBACK in {scope_label}",
                waitable=False,
                rule_id=scope_label,
                code="BLOCKED_INVALID_POLICY",
            )
        if allow_fb_val is True and explicit_strength == ConstraintStrength.HARD:
            raise PolicyBlockedError(
                f"BLOCKED_INVALID_POLICY: contradictory allow_fallback=True with {strength_key}=HARD in {scope_label}",
                waitable=False,
                rule_id=scope_label,
                code="BLOCKED_INVALID_POLICY",
            )
        return explicit_strength

    if implies_hard:
        return ConstraintStrength.HARD
    if allow_fb_val is True or forced_val is False or hard_val is False:
        return ConstraintStrength.SOFT_FALLBACK
    return fallback_strength


def _coerce_str_seq(raw_val: Any, *, key_name: str, scope_label: str) -> list[str]:
    if raw_val is None:
        return []
    if isinstance(raw_val, (str, bytes, Mapping)) or not isinstance(raw_val, Sequence):
        raise PolicyBlockedError(
            f"BLOCKED_INVALID_POLICY: {key_name!r} in {scope_label} must be a list of non-empty strings",
            waitable=False,
            rule_id=scope_label,
            code="BLOCKED_INVALID_POLICY",
        )
    out: list[str] = []
    for item in raw_val:
        if not isinstance(item, str) or not item.strip():
            raise PolicyBlockedError(
                f"BLOCKED_INVALID_POLICY: invalid entry {item!r} in {key_name!r} ({scope_label})",
                waitable=False,
                rule_id=scope_label,
                code="BLOCKED_INVALID_POLICY",
            )
        out.append(item.strip())
    return out


def _empty_disallowed_bundle() -> dict[str, list[str]]:
    return {"models": [], "families": [], "pools": [], "specs": []}


def _merge_disallowed_bundles(
    *bundles: Mapping[str, Sequence[str]] | None,
) -> dict[str, list[str]]:
    models: set[str] = set()
    families: set[str] = set()
    pools: set[str] = set()
    specs: set[str] = set()
    for b in bundles:
        if not isinstance(b, Mapping):
            continue
        models.update(str(x).lower() for x in (b.get("models") or ()))
        families.update(str(x).lower() for x in (b.get("families") or ()))
        pools.update(str(x).lower() for x in (b.get("pools") or ()))
        specs.update(str(x).lower() for x in (b.get("specs") or ()))
    return {
        "models": sorted(models),
        "families": sorted(families),
        "pools": sorted(pools),
        "specs": sorted(specs),
    }


def _normalize_custom_family_token(fam_raw: str, *, scope_label: str) -> str:
    fl = str(fam_raw or "").strip().lower()
    if fl in _KNOWN_FAMILY_TOKENS:
        return _KNOWN_FAMILY_TOKENS[fl]
    if fl.startswith("custom_") and len(fl) > 7:
        return fl
    raise PolicyBlockedError(
        f"BLOCKED_INVALID_POLICY: allowed_custom_commands in {scope_label} requires explicit family "
        f"(anthropic|openai|google|deepseek|xai|custom_<id>), got {fam_raw!r}",
        waitable=False,
        rule_id=scope_label,
        code="BLOCKED_INVALID_POLICY",
    )


def _normalize_allowed_custom_commands(
    raw_val: Any, *, scope_label: str
) -> dict[str, dict[str, str]]:
    if raw_val is None:
        return {}
    out: dict[str, dict[str, str]] = {}
    if isinstance(raw_val, Mapping):
        for k_raw, v_raw in raw_val.items():
            key_s = str(k_raw or "").strip()
            shell_cmd = key_s[4:].strip() if key_s.startswith("cmd:") else key_s
            if not shell_cmd:
                raise PolicyBlockedError(
                    f"BLOCKED_INVALID_POLICY: empty command key in allowed_custom_commands ({scope_label})",
                    waitable=False,
                    rule_id=scope_label,
                    code="BLOCKED_INVALID_POLICY",
                )
            if isinstance(v_raw, str):
                fam = _normalize_custom_family_token(v_raw, scope_label=scope_label)
            elif isinstance(v_raw, Mapping):
                fam = _normalize_custom_family_token(str(v_raw.get("family") or ""), scope_label=scope_label)
            else:
                raise PolicyBlockedError(
                    f"BLOCKED_INVALID_POLICY: invalid custom command registration {v_raw!r} in {scope_label}",
                    waitable=False,
                    rule_id=scope_label,
                    code="BLOCKED_INVALID_POLICY",
                )
            norm_key = f"cmd:{shell_cmd}"
            out[norm_key] = {"cmd": shell_cmd, "family": fam}
        return out

    if isinstance(raw_val, Sequence) and not isinstance(raw_val, (str, bytes)):
        for item in raw_val:
            if not isinstance(item, Mapping):
                raise PolicyBlockedError(
                    f"BLOCKED_INVALID_POLICY: allowed_custom_commands list item must be mapping in {scope_label}",
                    waitable=False,
                    rule_id=scope_label,
                    code="BLOCKED_INVALID_POLICY",
                )
            raw_cmd = str(item.get("spec") or item.get("cmd") or item.get("command") or "").strip()
            shell_cmd = raw_cmd[4:].strip() if raw_cmd.startswith("cmd:") else raw_cmd
            if not shell_cmd:
                raise PolicyBlockedError(
                    f"BLOCKED_INVALID_POLICY: empty command in allowed_custom_commands ({scope_label})",
                    waitable=False,
                    rule_id=scope_label,
                    code="BLOCKED_INVALID_POLICY",
                )
            fam = _normalize_custom_family_token(str(item.get("family") or ""), scope_label=scope_label)
            out[f"cmd:{shell_cmd}"] = {"cmd": shell_cmd, "family": fam}
        return out

    raise PolicyBlockedError(
        f"BLOCKED_INVALID_POLICY: allowed_custom_commands in {scope_label} must be a mapping or list",
        waitable=False,
        rule_id=scope_label,
        code="BLOCKED_INVALID_POLICY",
    )


def _merge_allowed_custom_commands(
    *maps: Mapping[str, Mapping[str, str]] | None,
) -> dict[str, dict[str, str]]:
    merged: dict[str, dict[str, str]] = {}
    for mp in maps:
        if not isinstance(mp, Mapping):
            continue
        for k, v in mp.items():
            if isinstance(v, Mapping) and v.get("family"):
                key_s = str(k).strip()
                norm_k = key_s if key_s.startswith("cmd:") else f"cmd:{key_s}"
                merged[norm_k] = {
                    "cmd": str(v.get("cmd") or norm_k[4:].strip()),
                    "family": str(v["family"]).strip().lower(),
                }
    return merged


def _parse_disallowed_model_or_spec_item(
    item: str,
    *,
    alias_index: Mapping[str, str],
    providers: Mapping[str, Mapping[str, Any]],
    scope_label: str,
) -> tuple[str | None, str | None]:
    """Return `(provider_key_lower, exact_spec_lower)` or raise `BLOCKED_INVALID_POLICY`."""
    low = item.strip().lower()
    if low.startswith("cmd:"):
        return low, low
    parts = [p.strip() for p in item.split(":") if p.strip()]
    if len(parts) == 1:
        m_gemini = _GEMINI_SUFFIX_RE.match(low)
        if m_gemini and low in alias_index:
            pkey = alias_index[low]
            return None, f"{pkey.lower()}:{m_gemini.group('eff').lower()}"
        if low in alias_index:
            return alias_index[low].lower(), None
        raise PolicyBlockedError(
            f"BLOCKED_INVALID_POLICY: unknown disallowed model {item!r} in {scope_label}",
            waitable=False,
            rule_id=scope_label,
            code="BLOCKED_INVALID_POLICY",
        )
    if len(parts) == 2:
        p0_low, p1_low = parts[0].lower(), parts[1].lower()
        if p1_low in alias_index and p1_low not in VALID_EFFORT_TOKENS:
            pkey = alias_index[p1_low]
            m_gemini = _GEMINI_SUFFIX_RE.match(p1_low)
            if m_gemini:
                return None, f"{pkey.lower()}:{m_gemini.group('eff').lower()}"
            return pkey.lower(), None
        if p0_low in alias_index and p1_low in VALID_EFFORT_TOKENS:
            pkey = alias_index[p0_low]
            eff = "extra high" if p1_low == "extra high" else p1_low
            return None, f"{pkey.lower()}:{eff}"
        raise PolicyBlockedError(
            f"BLOCKED_INVALID_POLICY: unknown disallowed model/spec {item!r} in {scope_label}",
            waitable=False,
            rule_id=scope_label,
            code="BLOCKED_INVALID_POLICY",
        )
    # 3+ parts: pool:model:effort
    p1_low, p2_low = parts[1].lower(), parts[2].lower()
    if p1_low not in alias_index or p2_low not in VALID_EFFORT_TOKENS:
        raise PolicyBlockedError(
            f"BLOCKED_INVALID_POLICY: unknown disallowed spec {item!r} in {scope_label}",
            waitable=False,
            rule_id=scope_label,
            code="BLOCKED_INVALID_POLICY",
        )
    pkey = alias_index[p1_low]
    _ = providers
    return None, f"{pkey.lower()}:{p2_low}"


def _normalize_disallowed_bundle(
    mapping: Mapping[str, Any],
    *,
    alias_index: Mapping[str, str],
    providers: Mapping[str, Mapping[str, Any]],
    scope_label: str,
) -> dict[str, list[str]]:
    models: set[str] = set()
    families: set[str] = set()
    pools: set[str] = set()
    specs: set[str] = set()

    existing_bundle = mapping.get("disallowed_bundle")
    if isinstance(existing_bundle, Mapping):
        merged_init = _merge_disallowed_bundles(existing_bundle)
        models.update(merged_init["models"])
        families.update(merged_init["families"])
        pools.update(merged_init["pools"])
        specs.update(merged_init["specs"])

    for fam_raw in _coerce_str_seq(mapping.get("disallowed_families"), key_name="disallowed_families", scope_label=scope_label):
        fl = fam_raw.lower()
        if fl in _KNOWN_FAMILY_TOKENS:
            families.add(_KNOWN_FAMILY_TOKENS[fl])
        elif fl.startswith("custom_"):
            families.add(fl)
        else:
            raise PolicyBlockedError(
                f"BLOCKED_INVALID_POLICY: unknown disallowed family {fam_raw!r} in {scope_label}",
                waitable=False,
                rule_id=scope_label,
                code="BLOCKED_INVALID_POLICY",
            )

    for pool_raw in _coerce_str_seq(mapping.get("disallowed_pools"), key_name="disallowed_pools", scope_label=scope_label):
        pl = pool_raw.lower()
        if pl in _KNOWN_POOL_TOKENS:
            pools.update(_KNOWN_POOL_TOKENS[pl])
        else:
            raise PolicyBlockedError(
                f"BLOCKED_INVALID_POLICY: unknown disallowed pool {pool_raw!r} in {scope_label}",
                waitable=False,
                rule_id=scope_label,
                code="BLOCKED_INVALID_POLICY",
            )

    for mod_raw in _coerce_str_seq(mapping.get("disallowed_models"), key_name="disallowed_models", scope_label=scope_label):
        m_key, s_key = _parse_disallowed_model_or_spec_item(
            mod_raw, alias_index=alias_index, providers=providers, scope_label=scope_label
        )
        if m_key:
            models.add(m_key)
        if s_key:
            specs.add(s_key)

    for gen_raw in _coerce_str_seq(mapping.get("disallowed"), key_name="disallowed", scope_label=scope_label):
        gl = gen_raw.lower()
        if ":" not in gl and gl in ("anthropic", "openai", "google", "deepseek", "xai"):
            families.add(gl)
        elif ":" not in gl and gl in _KNOWN_POOL_TOKENS and gl not in alias_index:
            pools.update(_KNOWN_POOL_TOKENS[gl])
        else:
            m_key, s_key = _parse_disallowed_model_or_spec_item(
                gen_raw, alias_index=alias_index, providers=providers, scope_label=scope_label
            )
            if m_key:
                models.add(m_key)
            if s_key:
                specs.add(s_key)

    return {
        "models": sorted(models),
        "families": sorted(families),
        "pools": sorted(pools),
        "specs": sorted(specs),
    }


def default_execution_host() -> str:
    """Logical execution host label used for provider host policies (`local` unless overridden)."""
    return str(os.environ.get("CONJECTRA_EXECUTION_HOST") or "local").strip().lower() or "local"


@functools.lru_cache(maxsize=8)
def _load_capabilities_cached(resolved_path: str) -> dict[str, Any]:
    path = Path(resolved_path)
    if not path.is_file():
        raise FileNotFoundError(f"PROVIDER_CAPABILITIES.json not found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_role(role: ScientificRole | str | None) -> str:
    """Normalize a ScientificRole enum or legacy role alias to ScientificRole.value."""
    if isinstance(role, ScientificRole):
        return role.value
    raw = str(role or ScientificRole.ENGINEER_BUILDER.value).strip().lower()
    return ROLE_ALIASES.get(raw, raw or ScientificRole.ENGINEER_BUILDER.value)


def _compute_rule_rank(roles: Sequence[str], tags: Sequence[str]) -> int:
    has_role = bool(roles)
    has_tag = bool(tags)
    if has_role and has_tag:
        return 2
    if has_tag and not has_role:
        return 3
    if has_role and not has_tag:
        return 4
    return 5


def _infer_custom_family(cmd_text: str) -> str:
    low = cmd_text.lower()
    for fam, markers in (
        ("anthropic", ("claude", "opus", "sonnet", "haiku")),
        ("openai", ("gpt", "codex")),
        ("google", ("gemini",)),
        ("deepseek", ("deepseek",)),
        ("xai", ("grok",)),
    ):
        if any(m in low for m in markers):
            return fam
    digest = hashlib.sha1(low.encode("utf-8")).hexdigest()[:6]
    return f"custom_{digest}"


def _build_alias_index(providers: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    idx: dict[str, str] = {}
    for pkey, rec in providers.items():
        idx[pkey.lower()] = pkey
        cmod = str(rec.get("canonical_model") or "").strip()
        if cmod:
            idx[cmod.lower()] = pkey
        dmod = str(rec.get("default_model_id") or "").strip()
        if dmod:
            idx[dmod.lower()] = pkey
        for al in rec.get("aliases") or []:
            if str(al).strip():
                idx[str(al).strip().lower()] = pkey
        for aid in (rec.get("effort_model_ids") or {}).values():
            if str(aid).strip():
                idx[str(aid).strip().lower()] = pkey
    return idx


def _resolve_provider_key_from_token(
    token: str,
    *,
    role_val: str,
    alias_index: Mapping[str, str],
) -> tuple[str | None, str | None]:
    """Resolve a raw model or pool token into `(provider_key, embedded_effort)`."""
    clean = token.strip()
    low = clean.lower()
    if not low:
        return None, None

    m_gemini = _GEMINI_SUFFIX_RE.match(low)
    embedded_eff = m_gemini.group("eff").lower() if m_gemini else None

    if low in alias_index:
        return alias_index[low], embedded_eff

    # Pool / HQ resource level tokens: never emit raw resource keys as model IDs!
    is_opus_role = role_val in (
        ScientificRole.RESEARCH_DIRECTOR.value,
        ScientificRole.THEORIST.value,
    )
    is_builder = role_val == ScientificRole.ENGINEER_BUILDER.value
    if low in ("claude", "llm_claude", "llmclaude"):
        return ("claude-opus-5-5" if is_opus_role else "claude-sonnet-5-5"), None
    if low in ("gemini", "gemini-cli", "llm_gemini", "llmgemini"):
        return ("gemini-flash" if is_builder else "gemini-pro"), None
    if low in ("codex", "llm_codex_audit", "llmcodexaudit"):
        return "gpt-5-codex", None
    if low in ("deepseek", "llm_deepseek", "llmdeepseek"):
        return "deepseek-chat", None
    if low == "grok":
        return "grok-code-fast", None
    if low in ("deepseek-local", "deepseek_local"):
        return "deepseek-local", None
    if low in ("webchat", "web_chat_audit"):
        return "webchat-gpt-xhigh", "Extra High"

    return None, None


class ModelRouter:
    """Single source of truth for provider capability validation, NL policy compilation, and routing."""

    def __init__(
        self,
        *,
        capabilities_path: Path | str | None = None,
        arbiter: ResourceArbiter | None = None,
        db: StateDB | None = None,
        foreground_reserve_quota_pct: float = 15.0,
        arr_home: Path | str | None = None,
    ) -> None:
        cap_p = Path(capabilities_path) if capabilities_path is not None else DEFAULT_CAPABILITIES_PATH
        self.capabilities_path = cap_p.expanduser().resolve()
        self.capabilities = _load_capabilities_cached(str(self.capabilities_path))
        self.providers: dict[str, dict[str, Any]] = {
            str(k): dict(v)
            for k, v in (self.capabilities.get("providers") or {}).items()
            if isinstance(v, Mapping)
        }
        self.hq_resource_mapping: dict[str, str] = {
            str(k): str(v)
            for k, v in (self.capabilities.get("hq_resource_mapping") or {}).items()
        }
        self.banned_channels: tuple[str, ...] = tuple(
            str(x).lower() for x in (self.capabilities.get("banned_channels") or ())
        )
        cap_custom = _normalize_allowed_custom_commands(
            self.capabilities.get("custom_commands"), scope_label="capabilities"
        )
        for pkey, prec in self.providers.items():
            if str(prec.get("pool") or "").strip().lower() == "cmd" and prec.get("family"):
                raw_c = str(prec.get("canonical_model") or pkey).strip()
                sc = raw_c[4:].strip() if raw_c.startswith("cmd:") else raw_c
                if sc:
                    cap_custom[f"cmd:{sc}"] = {
                        "cmd": sc,
                        "family": _normalize_custom_family_token(
                            str(prec["family"]), scope_label=f"capabilities.{pkey}"
                        ),
                    }
        self.custom_commands: dict[str, dict[str, str]] = cap_custom
        self.global_routing_policy: Any = self.capabilities.get("global_routing_policy")
        self._alias_index = _build_alias_index(self.providers)
        self.arbiter = arbiter
        self.db = db
        self.foreground_reserve_quota_pct = float(foreground_reserve_quota_pct)
        self.arr_home = Path(arr_home).expanduser().resolve() if arr_home is not None else None

    @staticmethod
    def compile_routing_policy(
        raw_policy: str | Mapping[str, Any] | ModelPolicy | None,
        *,
        project: str = "",
        capabilities_path: Path | str | None = None,
    ) -> dict[str, Any]:
        """Compile structured dict or Chinese/English NL policy into specificity-ranked rules.

        Raises ValueError on unknown/unparseable HARD model/effort clauses, banned channels,
        or conflicting HARD rules at the same specificity scope.
        """
        cap_p = Path(capabilities_path) if capabilities_path is not None else DEFAULT_CAPABILITIES_PATH
        cap_doc = _load_capabilities_cached(str(cap_p.expanduser().resolve()))
        providers = {
            str(k): dict(v)
            for k, v in (cap_doc.get("providers") or {}).items()
            if isinstance(v, Mapping)
        }
        banned_channels = tuple(str(x).lower() for x in (cap_doc.get("banned_channels") or ()))
        alias_index = _build_alias_index(providers)

        if raw_policy is None:
            return ModelPolicy(project=project).to_dict()

        if isinstance(raw_policy, ModelPolicy):
            raw_dict: Mapping[str, Any] = raw_policy.to_dict()
            return ModelRouter._compile_structured_mapping(
                raw_dict,
                project=project,
                providers=providers,
                alias_index=alias_index,
                banned_channels=banned_channels,
            )

        if isinstance(raw_policy, Mapping):
            return ModelRouter._compile_structured_mapping(
                raw_policy,
                project=project,
                providers=providers,
                alias_index=alias_index,
                banned_channels=banned_channels,
            )

        if not isinstance(raw_policy, str):
            raise ValueError(f"Unsupported routing policy type: {type(raw_policy).__name__}")

        nl_text = raw_policy.strip()
        if not nl_text:
            return ModelPolicy(project=project).to_dict()

        return ModelRouter._compile_nl_string(
            nl_text,
            project=project,
            providers=providers,
            alias_index=alias_index,
            banned_channels=banned_channels,
        )

    @staticmethod
    def _validate_candidate_spec_dict(
        cand: Mapping[str, Any] | str,
        *,
        strength: ConstraintStrength,
        providers: Mapping[str, Mapping[str, Any]],
        alias_index: Mapping[str, str],
        banned_channels: Sequence[str],
    ) -> dict[str, str]:
        if isinstance(cand, str):
            raw_spec = cand.strip()
            effort_override = ""
        elif isinstance(cand, Mapping):
            _check_allowed_mapping_keys(cand, _ALLOWED_CANDIDATE_KEYS, scope_label="candidate")
            raw_spec = str(
                cand.get("spec")
                or cand.get("provider")
                or cand.get("provider_key")
                or cand.get("model")
                or cand.get("canonical_model")
                or ""
            ).strip()
            effort_override = str(cand.get("effort") or "").strip()
        else:
            raise PolicyBlockedError(
                f"BLOCKED_INVALID_POLICY: invalid routing candidate {cand!r}",
                waitable=False,
                code="BLOCKED_INVALID_POLICY",
            )

        low_spec = raw_spec.lower()
        for banned in banned_channels:
            if banned and banned in low_spec:
                raise ValueError(f"Banned channel {banned!r} in routing policy candidate {raw_spec!r}")

        if raw_spec.startswith("cmd:"):
            out_cmd: dict[str, str] = {"provider_key": raw_spec, "effort": "custom"}
            if isinstance(cand, Mapping) and cand.get("family"):
                out_cmd["family"] = _normalize_custom_family_token(
                    str(cand["family"]), scope_label="candidate.cmd"
                )
            return out_cmd

        parts = [p.strip() for p in raw_spec.split(":") if p.strip()]
        if not parts:
            raise ValueError(f"Empty model candidate in routing policy: {cand!r}")

        if len(parts) == 1:
            model_tok = parts[0]
            eff_tok = effort_override
        elif len(parts) == 2:
            # Could be `pool:model` or `model:effort`
            p0_prov, _ = _resolve_provider_key_from_token(
                parts[0], role_val="research_lead", alias_index=alias_index
            )
            p1_prov, p1_emb = _resolve_provider_key_from_token(
                parts[1], role_val="research_lead", alias_index=alias_index
            )
            if p1_prov is not None:
                model_tok = parts[1]
                eff_tok = effort_override or (p1_emb or "")
            elif parts[1].lower() in VALID_EFFORT_TOKENS or parts[1].lower() in UNKNOWN_EFFORT_TRAP_TOKENS:
                model_tok = parts[0]
                eff_tok = effort_override or parts[1]
            else:
                model_tok = parts[1]
                eff_tok = effort_override
                _ = p0_prov
        else:
            model_tok = parts[1]
            eff_tok = effort_override or parts[2]

        pkey, emb_eff = _resolve_provider_key_from_token(
            model_tok, role_val="research_lead", alias_index=alias_index
        )
        if pkey is None:
            if strength == ConstraintStrength.HARD:
                raise ValueError(f"Unknown model {model_tok!r} in HARD routing rule ({raw_spec!r})")
            raise ValueError(f"Unknown model {model_tok!r} in routing candidate ({raw_spec!r})")

        norm_eff = (eff_tok or emb_eff or str(providers[pkey].get("default_effort") or "high")).strip()
        low_eff = norm_eff.lower()
        if low_eff == "extra high":
            norm_eff = "Extra High" if pkey == "webchat-gpt-xhigh" else "xhigh"
            low_eff = norm_eff.lower()
        if low_eff in UNKNOWN_EFFORT_TRAP_TOKENS or low_eff not in VALID_EFFORT_TOKENS:
            raise ValueError(f"Unknown effort level {norm_eff!r} for model {pkey!r}")

        return {
            "provider_key": pkey,
            "effort": norm_eff if norm_eff == "Extra High" else low_eff,
        }

    @staticmethod
    def _check_hard_rule_conflicts(rules: Sequence[Mapping[str, Any]]) -> None:
        seen_hard: dict[tuple[Any, ...], tuple[str, tuple[tuple[str, str], ...]]] = {}
        for rule in rules:
            if str(rule.get("strength")) != ConstraintStrength.HARD.value:
                continue
            scope_key = (
                int(rule.get("rank", 5)),
                tuple(sorted(str(r) for r in (rule.get("roles") or []))),
                tuple(sorted(str(t) for t in (rule.get("tags") or []))),
                bool(rule.get("author_only", False)),
                str(rule.get("project_scope") or ""),
            )
            cand_sig = tuple(
                (str(c.get("provider_key")), str(c.get("effort")))
                for c in (rule.get("candidates") or [])
            )
            prev = seen_hard.get(scope_key)
            if prev is not None and prev[1] != cand_sig:
                raise ValueError(
                    f"Conflicting HARD routing rules {prev[0]!r} ({prev[1]}) and "
                    f"{rule.get('rule_id')!r} ({cand_sig}) at identical specificity scope {scope_key}"
                )
            seen_hard[scope_key] = (str(rule.get("rule_id") or "rule"), cand_sig)

    @staticmethod
    def _compile_structured_mapping(
        raw: Mapping[str, Any],
        *,
        project: str,
        providers: Mapping[str, Mapping[str, Any]],
        alias_index: Mapping[str, str],
        banned_channels: Sequence[str],
    ) -> dict[str, Any]:
        _check_allowed_mapping_keys(raw, _ALLOWED_POLICY_KEYS, scope_label="routing_policy")
        source_nl = str(raw.get("source_nl") or raw.get("routing_policy_nl") or "").strip()
        eff_project = str(raw.get("project") or project or "").strip()
        default_strength = _normalize_strength_from_mapping(
            raw,
            fallback_strength=ConstraintStrength.SOFT_FALLBACK,
            scope_label="routing_policy",
            strength_key="default_strength",
        )
        policy_implies_hard = (
            raw.get("forced") is True
            or raw.get("hard") is True
            or raw.get("allow_fallback") is False
        )
        allow_effort_alias = bool(raw.get("allow_effort_alias", False))
        reserve_pct = float(raw.get("foreground_reserve_quota_pct", 15.0))
        policy_disallowed = _normalize_disallowed_bundle(
            raw,
            alias_index=alias_index,
            providers=providers,
            scope_label="routing_policy",
        )
        allowed_custom_cmds = _normalize_allowed_custom_commands(
            raw.get("allowed_custom_commands"),
            scope_label="routing_policy",
        )

        raw_rules_val = raw.get("rules")
        if raw_rules_val is not None and (
            isinstance(raw_rules_val, (str, bytes, Mapping)) or not isinstance(raw_rules_val, Sequence)
        ):
            raise PolicyBlockedError(
                "BLOCKED_INVALID_POLICY: 'rules' in routing_policy must be a list of rule mappings",
                waitable=False,
                rule_id="routing_policy",
                code="BLOCKED_INVALID_POLICY",
            )
        raw_rules = list(raw_rules_val or [])
        review_overrides = {
            str(k).strip().lower(): str(v).strip().upper()
            for k, v in dict(raw.get("review_overrides") or {}).items()
            if str(k).strip() and str(v).strip()
        }

        if not raw_rules and source_nl:
            compiled_nl = ModelRouter._compile_nl_string(
                source_nl,
                project=eff_project,
                providers=providers,
                alias_index=alias_index,
                banned_channels=banned_channels,
            )
            compiled_nl["review_overrides"].update(review_overrides)
            if "allow_effort_alias" in raw:
                compiled_nl["allow_effort_alias"] = allow_effort_alias
            if "foreground_reserve_quota_pct" in raw:
                compiled_nl["foreground_reserve_quota_pct"] = reserve_pct
            if any(k in raw for k in ("default_strength", "forced", "hard", "allow_fallback")):
                compiled_nl["default_strength"] = default_strength.value
            if policy_implies_hard:
                for nl_r in compiled_nl.get("rules") or []:
                    if isinstance(nl_r, dict):
                        nl_r["strength"] = ConstraintStrength.HARD.value
            compiled_nl["disallowed_bundle"] = policy_disallowed
            compiled_nl["allowed_custom_commands"] = allowed_custom_cmds
            compiled_nl["has_active_constraints"] = True
            return compiled_nl

        normalized_rules: list[dict[str, Any]] = []
        for idx, r in enumerate(raw_rules):
            if not isinstance(r, Mapping):
                raise PolicyBlockedError(
                    f"BLOCKED_INVALID_POLICY: routing rule at index {idx} must be a mapping, got {r!r}",
                    waitable=False,
                    rule_id=f"rule_{idx + 1}",
                    code="BLOCKED_INVALID_POLICY",
                )
            r_label = str(r.get("rule_id") or f"rule_{idx + 1}")
            _check_allowed_mapping_keys(r, _ALLOWED_RULE_KEYS, scope_label=r_label)
            r_strength = _normalize_strength_from_mapping(
                r,
                fallback_strength=default_strength,
                scope_label=r_label,
                strength_key="strength",
            )
            r_disallowed = _normalize_disallowed_bundle(
                r,
                alias_index=alias_index,
                providers=providers,
                scope_label=r_label,
            )
            roles_in = r.get("roles")
            if roles_in is None and r.get("role"):
                roles_in = [r["role"]]
            norm_roles = sorted(
                {normalize_role(str(x)) for x in (roles_in or []) if str(x).strip()}
            )

            tags_in = r.get("tags")
            if tags_in is None and r.get("tag"):
                tags_in = [r["tag"]]
            norm_tags = sorted(
                {str(x).strip().lower() for x in (tags_in or []) if str(x).strip()}
            )

            raw_cands = list(r.get("candidates") or [])
            if not raw_cands:
                single = r.get("spec") or r.get("model") or r.get("provider")
                if single:
                    raw_cands = [{"spec": str(single), "effort": str(r.get("effort") or "")}]
            if not raw_cands:
                raise PolicyBlockedError(
                    f"BLOCKED_INVALID_POLICY: routing rule {r_label!r} has zero model candidates",
                    waitable=False,
                    rule_id=r_label,
                    code="BLOCKED_INVALID_POLICY",
                )

            cands = [
                ModelRouter._validate_candidate_spec_dict(
                    c,
                    strength=r_strength,
                    providers=providers,
                    alias_index=alias_index,
                    banned_channels=banned_channels,
                )
                for c in raw_cands
            ]
            for c_item in cands:
                pk_c = str(c_item.get("provider_key") or "")
                if pk_c.startswith("cmd:") and c_item.get("family"):
                    sc_c = pk_c[4:].strip()
                    if sc_c:
                        allowed_custom_cmds.setdefault(
                            f"cmd:{sc_c}", {"cmd": sc_c, "family": str(c_item["family"]).lower()}
                        )
            rank = _compute_rule_rank(norm_roles, norm_tags)
            author_only = bool(
                r.get(
                    "author_only",
                    bool(norm_tags) and not any(rv in REVIEWER_ROLE_VALUES for rv in norm_roles),
                )
            )
            rule_entry: dict[str, Any] = {
                "rule_id": r_label,
                "rank": rank,
                "roles": norm_roles,
                "tags": norm_tags,
                "candidates": cands,
                "strength": r_strength.value,
                "allow_effort_alias": bool(r.get("allow_effort_alias", allow_effort_alias)),
                "author_only": author_only,
                "project_scope": str(r.get("project_scope") or "").strip().lower(),
                "disallowed_bundle": r_disallowed,
            }
            if "_scope_level" in r:
                rule_entry["_scope_level"] = int(r["_scope_level"])
            if "_scope_name" in r:
                rule_entry["_scope_name"] = str(r["_scope_name"])
            normalized_rules.append(rule_entry)

        ModelRouter._check_hard_rule_conflicts(normalized_rules)
        normalized_rules.sort(key=lambda item: int(item["rank"]))

        compiled_out = ModelPolicy(
            source_nl=source_nl,
            project=eff_project,
            rules=normalized_rules,
            review_overrides=review_overrides,
            default_strength=default_strength,
            allow_effort_alias=allow_effort_alias,
            foreground_reserve_quota_pct=reserve_pct,
        ).to_dict()
        compiled_out["disallowed_bundle"] = policy_disallowed
        compiled_out["allowed_custom_commands"] = allowed_custom_cmds
        has_any_disallow = any(bool(v) for v in policy_disallowed.values())
        compiled_out["has_active_constraints"] = bool(
            normalized_rules
            or source_nl
            or has_any_disallow
            or review_overrides
            or allowed_custom_cmds
            or default_strength == ConstraintStrength.HARD
            or policy_implies_hard
            or bool(raw.get("has_active_constraints", False))
        )
        return compiled_out

    @staticmethod
    def _compile_nl_string(
        nl_text: str,
        *,
        project: str,
        providers: Mapping[str, Mapping[str, Any]],
        alias_index: Mapping[str, str],
        banned_channels: Sequence[str],
    ) -> dict[str, Any]:
        low_full = nl_text.lower()
        for banned in banned_channels:
            if banned and banned in low_full:
                raise ValueError(f"Banned channel {banned!r} referenced in routing policy: {nl_text!r}")

        clauses = [
            c.strip()
            for c in re.split(r"[，,；;。\n]+", nl_text)
            if c and c.strip()
        ]
        rules: list[dict[str, Any]] = []
        review_overrides: dict[str, str] = {}

        for idx, clause in enumerate(clauses, start=1):
            c_low = clause.lower()

            # 1. Detect cross-family / red-team review requirement clauses
            is_cross_family_review_clause = any(
                kw in c_low
                for kw in (
                    "另一模型家族",
                    "跨家族审查",
                    "异构审查",
                    "独立模型家族",
                    "cross-family",
                    "another model family",
                    "different model family",
                    "red-team review",
                )
            ) or (("审查" in c_low or "review" in c_low) and ("家族" in c_low or "family" in c_low))

            clause_tags: list[str] = []
            if any(k in c_low for k in ("理论", "定理", "推导", "theory", "theorem", "proof")):
                clause_tags.append("theory")
            if any(k in c_low for k in ("backbone", "主干", "架构", "architecture")):
                clause_tags.append("backbone")
            if any(k in c_low for k in ("里程碑", "结项", "milestone", "closeout")):
                clause_tags.append("milestone")
            if any(k in c_low for k in ("消融", "ablation")):
                clause_tags.append("ablation")

            if is_cross_family_review_clause:
                tier_val = (
                    ReviewTier.RED_TEAM_TIEBREAKER.value
                    if (
                        any(k in c_low for k in ("重大", "必须", "严格", "major", "critical", "must", "red-team"))
                        or bool(set(clause_tags) & {"theory", "backbone", "milestone"})
                    )
                    else ReviewTier.STANDARD_CROSS_FAMILY.value
                )
                for tg in (clause_tags or ["*"]):
                    review_overrides[tg] = tier_val

            # 2. Extract model + effort candidates in appearance order
            matched_spans: list[tuple[int, int]] = []
            extracted_cands: list[dict[str, str]] = []

            for pat, pkey in _NL_MODEL_PATTERNS:
                for m in pat.finditer(clause):
                    s_pos, e_pos = m.span()
                    if any(not (e_pos <= a or s_pos >= b) for a, b in matched_spans):
                        continue
                    matched_spans.append((s_pos, e_pos))
                    matched_tok = m.group(0)
                    m_gemini = _GEMINI_SUFFIX_RE.match(matched_tok.lower())
                    emb_eff = m_gemini.group("eff").lower() if m_gemini else None

                    tail = clause[e_pos:]
                    m_eff = _TRAILING_EFFORT_RE.match(tail)
                    if m_eff:
                        raw_eff = m_eff.group(1).strip()
                        if raw_eff.lower() in UNKNOWN_EFFORT_TRAP_TOKENS:
                            raise ValueError(
                                f"Unknown effort token {raw_eff!r} in policy clause {clause!r}"
                            )
                        eff_val = (
                            "Extra High"
                            if (raw_eff.lower() == "extra high" and pkey == "webchat-gpt-xhigh")
                            else ("xhigh" if raw_eff.lower() == "extra high" else raw_eff.lower())
                        )
                    else:
                        # Reject explicit malformed effort syntax (`:bogus` / `=bogus`) or known trap effort modifiers
                        m_explicit_bad = re.match(r"^\s*[:=]\s*([a-zA-Z0-9_-]{1,20})\b", tail)
                        if m_explicit_bad and m_explicit_bad.group(1).lower() not in VALID_EFFORT_TOKENS:
                            raise ValueError(
                                f"Unknown effort modifier {m_explicit_bad.group(1)!r} after {matched_tok!r} in {clause!r}"
                            )
                        m_trap = re.match(r"^\s+([a-zA-Z][a-zA-Z0-9_-]{1,15})\b", tail)
                        if m_trap and m_trap.group(1).lower() in UNKNOWN_EFFORT_TRAP_TOKENS:
                            raise ValueError(
                                f"Unknown effort modifier {m_trap.group(1)!r} after {matched_tok!r} in {clause!r}"
                            )
                        eff_val = emb_eff or str(providers[pkey].get("default_effort") or "high")

                    extracted_cands.append(
                        {
                            "_pos": str(s_pos),
                            "provider_key": pkey,
                            "effort": eff_val,
                        }
                    )

            extracted_cands.sort(key=lambda item: int(item.pop("_pos")))

            if any(k in c_low for k in ("低成本模型", "轻量模型", "low-cost model", "low cost model", "cheap model")):
                for fallback_pk, fallback_eff in (
                    ("deepseek-chat", "high"),
                    ("gemini-flash", "high"),
                    ("claude-sonnet-5-5", "medium"),
                ):
                    if not any(c["provider_key"] == fallback_pk for c in extracted_cands):
                        extracted_cands.append({"provider_key": fallback_pk, "effort": fallback_eff})

            if not extracted_cands:
                if is_cross_family_review_clause:
                    continue
                if any(
                    k in c_low
                    for k in ("使用", "必须", "强制", "仅限", "must use", "use ", "require ", "only use")
                ):
                    raise ValueError(
                        f"Unparseable model/effort routing clause (no known provider matched): {clause!r}"
                    )
                continue

            # 3. Extract role scope
            clause_roles: list[str] = []
            if any(
                k in c_low
                for k in ("实现审查", "代码审查", "impl review", "implementation review", "code review", "adversary")
            ):
                clause_roles.append(ScientificRole.IMPL_REVIEWER.value)
            if any(k in c_low for k in ("设计审查", "方案审查", "design review")):
                clause_roles.append(ScientificRole.DESIGN_REVIEWER.value)
            if any(k in c_low for k in ("结果审查", "结论审查", "盲审", "result review", "result audit", "blind audit")):
                clause_roles.append(ScientificRole.RESULT_AUDITOR.value)
            if not clause_roles and any(k in c_low for k in ("审查", "审计", "review", "auditor")):
                if not is_cross_family_review_clause:
                    clause_roles.extend(sorted(REVIEWER_ROLE_VALUES))

            if any(
                k in c_low
                for k in ("普通实现", "常规实现", "代码实现", "工程实现", "实现使用", "implementation", "builder", "coding")
            ):
                clause_roles.append(ScientificRole.ENGINEER_BUILDER.value)
            if any(
                k in c_low
                for k in ("科研分析", "常规分析", "实验设计", "分析使用", "research analysis", "routine analysis", "experiment design")
            ):
                clause_roles.extend(
                    [
                        ScientificRole.RESEARCH_LEAD.value,
                        ScientificRole.EXPERIMENT_DESIGNER.value,
                    ]
                )
            if any(k in c_low for k in ("理论家", "theorist")):
                clause_roles.append(ScientificRole.THEORIST.value)
            if any(k in c_low for k in ("研究总监", "总监", "director", "adjudicat")):
                clause_roles.append(ScientificRole.RESEARCH_DIRECTOR.value)

            norm_roles = sorted(set(clause_roles))
            norm_tags = sorted(set(clause_tags))

            # 4. Infer HARD vs SOFT_FALLBACK strength
            has_hard_kw = any(
                k in c_low
                for k in ("必须", "强制", "仅限", "严格", "只允许", "must", "strictly", "only use", "forbid")
            )
            has_soft_kw = any(
                k in c_low
                for k in (
                    "允许",
                    "可以",
                    "优先",
                    "普通",
                    "常规",
                    "默认",
                    "或",
                    "allow",
                    "may",
                    "prefer",
                    "routine",
                    "fallback",
                    "default",
                    "unless",
                    "otherwise",
                    "if available",
                    "when available",
                    " or ",
                )
            )
            if has_hard_kw:
                strength = ConstraintStrength.HARD
            elif has_soft_kw or len(extracted_cands) > 1:
                strength = ConstraintStrength.SOFT_FALLBACK
            else:
                strength = ConstraintStrength.HARD

            allow_alias = any(k in c_low for k in ("允许降级", "允许映射", "allow alias", "allow_effort_alias"))
            project_scope = ""
            m_proj = re.search(r"\bproject\s*[:=]\s*([a-z0-9][a-z0-9_.-]*)", c_low)
            if m_proj:
                project_scope = m_proj.group(1)

            rank = _compute_rule_rank(norm_roles, norm_tags)
            author_only = not any(rv in REVIEWER_ROLE_VALUES for rv in norm_roles)

            rules.append(
                {
                    "rule_id": f"nl_clause_{idx}",
                    "rank": rank,
                    "roles": norm_roles,
                    "tags": norm_tags,
                    "candidates": extracted_cands,
                    "strength": strength.value,
                    "allow_effort_alias": allow_alias,
                    "author_only": author_only,
                    "project_scope": project_scope,
                    "clause": clause,
                }
            )

        ModelRouter._check_hard_rule_conflicts(rules)
        rules.sort(key=lambda item: int(item["rank"]))
        _ = alias_index
        return ModelPolicy(
            source_nl=nl_text,
            project=project,
            rules=rules,
            review_overrides=review_overrides,
            default_strength=ConstraintStrength.SOFT_FALLBACK,
            allow_effort_alias=False,
            foreground_reserve_quota_pct=15.0,
        ).to_dict()

    def _default_candidates_for_role(self, role_val: str) -> list[dict[str, str]]:
        if role_val in (ScientificRole.RESEARCH_DIRECTOR.value, ScientificRole.THEORIST.value):
            return [
                {"provider_key": "claude-opus-5-5", "effort": "high"},
                {"provider_key": "gpt-5-pro", "effort": "high"},
                {"provider_key": "gemini-pro", "effort": "high"},
            ]
        if role_val in (ScientificRole.RESEARCH_LEAD.value, ScientificRole.EXPERIMENT_DESIGNER.value):
            return [
                {"provider_key": "gpt-5-codex", "effort": "high"},
                {"provider_key": "gemini-flash", "effort": "high"},
                {"provider_key": "gemini-pro", "effort": "high"},
                {"provider_key": "claude-sonnet-5-5", "effort": "medium"},
            ]
        if role_val in REVIEWER_ROLE_VALUES:
            return [
                {"provider_key": "gpt-5-codex", "effort": "high"},
                {"provider_key": "gemini-pro", "effort": "high"},
                {"provider_key": "grok-code-fast", "effort": "high"},
                {"provider_key": "claude-sonnet-5-5", "effort": "medium"},
                {"provider_key": "deepseek-chat", "effort": "high"},
            ]
        return [
            {"provider_key": "claude-sonnet-5-5", "effort": "medium"},
            {"provider_key": "gemini-pro", "effort": "high"},
            {"provider_key": "deepseek-chat", "effort": "high"},
            {"provider_key": "gpt-5-codex", "effort": "high"},
            {"provider_key": "gemini-flash", "effort": "high"},
        ]

    def _select_backend_mode(self, pool: str, host: str) -> str:
        if pool == "webchat":
            return "webchat"
        forced = str(os.environ.get("CONJECTRA_BACKEND_OVERRIDE") or "").strip()
        if forced:
            return forced
        explicit = str((self.providers.get(pool) or {}).get("backend") or "").strip()
        if explicit:
            return explicit
        if pool == "fake":
            return "fake"
        if host == "local":
            if pool == "claude":
                return "claude_cli"
            if pool == "codex":
                return "codex_cli"
            if pool == "gemini":
                return "gemini_cli"
            if pool in ("deepseek", "grok"):
                return "openai_compat"
        return "dispatch_script"

    def _evaluate_candidate(
        self,
        *,
        provider_key: str,
        requested_effort: str,
        role_val: str,
        host: str,
        strength: ConstraintStrength,
        allow_effort_alias: bool,
        rule_id: str,
        avoid_set: set[str],
        excluded_set: set[str],
        headroom: Mapping[str, Mapping[str, Any]],
        active_pings: Mapping[str, Mapping[str, Any]],
        reserve_pct: float,
        force_multipool: bool = False,
        disallowed_bundle: Mapping[str, Sequence[str]] | None = None,
        has_active_policy: bool = False,
        allowed_custom_commands: Mapping[str, Mapping[str, str]] | None = None,
        candidate_family: str = "",
    ) -> tuple[ResolvedModelRoute | None, PolicyBlockedError | None]:
        dis_models = {str(x).lower() for x in ((disallowed_bundle or {}).get("models") or ())}
        dis_families = {str(x).lower() for x in ((disallowed_bundle or {}).get("families") or ())}
        dis_pools = {str(x).lower() for x in ((disallowed_bundle or {}).get("pools") or ())}
        dis_specs = {str(x).lower() for x in ((disallowed_bundle or {}).get("specs") or ())}

        if provider_key.startswith("cmd:"):
            shell_cmd = provider_key[4:].strip()
            if not shell_cmd:
                return None, PolicyBlockedError("Empty custom command in cmd: spec", rule_id=rule_id)
            exe_name = Path(shlex.split(shell_cmd)[0]).name
            norm_cmd_key = f"cmd:{shell_cmd}"
            reg_map = dict(allowed_custom_commands or {})
            reg_entry = reg_map.get(norm_cmd_key)
            if reg_entry is None:
                for rk_cand, rv_cand in reg_map.items():
                    if rk_cand.lower() == norm_cmd_key.lower():
                        reg_entry = rv_cand
                        break
            if has_active_policy and reg_entry is None and not candidate_family:
                return None, PolicyBlockedError(
                    f"BLOCKED_UNREGISTERED_CUSTOM_CMD: custom command {provider_key!r} is forbidden in "
                    "policy-bound runs unless registered in capabilities/allowed_custom_commands with declared family",
                    waitable=False,
                    rule_id=rule_id,
                    code="BLOCKED_UNREGISTERED_CUSTOM_CMD",
                )
            if reg_entry is not None and reg_entry.get("family"):
                fam = str(reg_entry["family"]).strip().lower()
            elif candidate_family:
                fam = str(candidate_family).strip().lower()
            else:
                fam = _infer_custom_family(shell_cmd)
            low_pk = norm_cmd_key.lower()
            if (
                fam in dis_families
                or "cmd" in dis_pools
                or f"cmd:{exe_name}".lower() in dis_pools
                or low_pk in dis_models
                or low_pk in dis_specs
            ):
                return None, PolicyBlockedError(
                    f"BLOCKED_DISALLOWED: custom command {provider_key!r} is disallowed by policy",
                    waitable=False,
                    rule_id=rule_id,
                    code="BLOCKED_DISALLOWED",
                )
            if fam in avoid_set:
                return None, PolicyBlockedError(
                    f"BLOCKED_CROSS_FAMILY: custom command family {fam!r} in avoid_families={sorted(avoid_set)}",
                    waitable=False,
                    rule_id=rule_id,
                    code="BLOCKED_CROSS_FAMILY",
                )
            return (
                ResolvedModelRoute(
                    backend_mode="custom_cmd",
                    execution_host=host,
                    multipool_host_arg=host,
                    pool="cmd",
                    canonical_model=shell_cmd,
                    requested_effort="custom",
                    wire_effort="custom",
                    family=fam,
                    logical_pool=f"cmd:{exe_name}",
                    hq_resource=None,
                    strength=strength,
                    matched_rule_id=rule_id,
                    allow_effort_alias=allow_effort_alias,
                ),
                None,
            )

        prov = self.providers.get(provider_key)
        if prov is None:
            return None, PolicyBlockedError(
                f"BLOCKED_POLICY: unknown provider {provider_key!r}",
                waitable=False,
                rule_id=rule_id,
            )

        if prov.get("enabled_by_default") is False and rule_id != "explicit_spec":
            return None, PolicyBlockedError(
                f"BLOCKED_POLICY: provider {provider_key!r} is disabled by default",
                waitable=False,
                rule_id=rule_id,
            )

        pool = str(prov["pool"])
        family = str(prov["family"]).lower()
        logical_pool = str(prov.get("logical_pool") or "")
        raw_hq_res = prov.get("hq_resource")
        hq_resource = None if raw_hq_res is None else str(raw_hq_res)

        if family in avoid_set:
            return None, PolicyBlockedError(
                f"BLOCKED_CROSS_FAMILY: role={role_val} candidate {provider_key!r} (family={family!r}) "
                f"overlaps avoid_families={sorted(avoid_set)}",
                waitable=False,
                rule_id=rule_id,
                code="BLOCKED_CROSS_FAMILY",
            )

        allowed_hosts = [str(h).lower() for h in (prov.get("allowed_execution_hosts") or [])]
        blocked_hosts = [str(h).lower() for h in (prov.get("blocked_execution_hosts") or [])]
        if host in blocked_hosts or (allowed_hosts and host not in allowed_hosts):
            code = "BLOCKED_HOST"
            return None, PolicyBlockedError(
                f"{code}: provider {provider_key!r} cannot execute on host={host!r}",
                waitable=False,
                rule_id=rule_id,
                code=code,
            )

        backend_mode = "dispatch_script" if force_multipool else self._select_backend_mode(pool, host)
        mp_host_arg = str(prov.get("dispatch_host_arg") or host)

        req_eff = (
            requested_effort
            if requested_effort == "Extra High"
            else requested_effort.strip().lower()
        )
        if pool == "webchat" and req_eff in ("xhigh", "extra high", "high"):
            req_eff = "Extra High" if req_eff in ("xhigh", "extra high") else "High"

        # Global Sonnet cap: sonnet > medium is strictly forbidden under HARD even if allow_effort_alias=True
        if provider_key == "claude-sonnet-5-5" and req_eff in ("high", "xhigh", "max"):
            if strength == ConstraintStrength.HARD:
                return None, PolicyBlockedError(
                    f"BLOCKED_SONNET_CAP: claude-sonnet-5-5 strictly forbids effort={req_eff!r} (<=medium only)",
                    waitable=False,
                    rule_id=rule_id,
                    code="BLOCKED_SONNET_CAP",
                )
            wire_effort = "medium"
        else:
            policy_efforts = set(prov.get("policy_allowed_efforts") or [])
            backend_efforts = set(
                prov.get("multipool_argparse_efforts") or []
                if backend_mode == "dispatch_script"
                else (prov.get("native_cli_efforts") or [])
            )
            exact_allowed = policy_efforts & backend_efforts
            if req_eff in exact_allowed:
                wire_effort = req_eff
            else:
                if strength == ConstraintStrength.HARD and not allow_effort_alias:
                    return None, PolicyBlockedError(
                        f"BLOCKED_POLICY: {provider_key} on {backend_mode} does not natively support "
                        f"exact effort={req_eff!r} (allowed={sorted(exact_allowed)})",
                        waitable=False,
                        rule_id=rule_id,
                        code="BLOCKED_POLICY",
                    )
                effort_map = dict(prov.get("multipool_effort_map") or {})
                mapped_eff = effort_map.get(req_eff)
                if not mapped_eff or mapped_eff not in backend_efforts:
                    return None, PolicyBlockedError(
                        f"BLOCKED_POLICY: {provider_key} has no verified wire effort mapping for {req_eff!r}",
                        waitable=False,
                        rule_id=rule_id,
                        code="BLOCKED_POLICY",
                    )
                wire_effort = str(mapped_eff)

        base_cmod = str(prov["canonical_model"])
        effort_ids = prov.get("effort_model_ids")
        if isinstance(effort_ids, Mapping) and effort_ids.get(wire_effort):
            # Some CLIs encode reasoning effort in the model id itself (e.g. `<model>-low`).
            canonical_model = str(effort_ids[wire_effort])
        else:
            canonical_model = base_cmod

        if canonical_model.lower() in _RAW_HQ_RESOURCE_NAMES:
            return None, PolicyBlockedError(
                f"BLOCKED_POLICY: invalid raw HQ resource name {canonical_model!r} as canonical_model",
                waitable=False,
                rule_id=rule_id,
            )

        cand_full_spec = f"{pool}:{canonical_model}:{req_eff}".lower()
        cand_short_spec = f"{pool}:{canonical_model}".lower()
        cand_wire_spec = f"{pool}:{canonical_model}:{wire_effort}".lower()
        pk_req_spec = f"{provider_key}:{req_eff}".lower()
        pk_wire_spec = f"{provider_key}:{wire_effort}".lower()

        if (
            family in dis_families
            or pool.lower() in dis_pools
            or (logical_pool and logical_pool.lower() in dis_pools)
            or (hq_resource and hq_resource.lower() in dis_pools)
            or provider_key.lower() in dis_models
            or canonical_model.lower() in dis_models
            or cand_full_spec in dis_specs
            or cand_wire_spec in dis_specs
            or pk_req_spec in dis_specs
            or pk_wire_spec in dis_specs
        ):
            return None, PolicyBlockedError(
                f"BLOCKED_DISALLOWED: {provider_key} ({cand_full_spec}) is disallowed by routing policy",
                waitable=False,
                rule_id=rule_id,
                code="BLOCKED_DISALLOWED",
            )

        if (
            cand_full_spec in excluded_set
            or cand_short_spec in excluded_set
            or cand_wire_spec in excluded_set
            or canonical_model.lower() in excluded_set
            or provider_key.lower() in excluded_set
        ):
            return None, PolicyBlockedError(
                f"BLOCKED_EXCLUDED_SPEC: {cand_full_spec} is excluded for this attempt",
                waitable=False,
                rule_id=rule_id,
                code="BLOCKED_EXCLUDED_SPEC",
            )

        # Valley CNY guard if deepseek-local was explicitly requested
        if logical_pool == "deepseek_local" and self.arbiter is not None:
            cny = self.arbiter.read_cny_spend_today()
            if not cny.get("valley_allowed", False):
                return None, PolicyBlockedError(
                    f"BLOCKED_QUOTA: deepseek_local fuse blocked (spent={cny.get('spent_cny')})",
                    waitable=False,
                    rule_id=rule_id,
                    code="BLOCKED_QUOTA",
                )

        tentative_route = ResolvedModelRoute(
            backend_mode=backend_mode,
            execution_host=host,
            multipool_host_arg=mp_host_arg,
            pool=pool,
            canonical_model=canonical_model,
            requested_effort=req_eff,
            wire_effort=wire_effort,
            family=family,
            logical_pool=logical_pool,
            hq_resource=hq_resource,
            strength=strength,
            matched_rule_id=rule_id,
            allow_effort_alias=allow_effort_alias,
        )

        # Live 7d quota & foreground interactive reserve checks
        pool_info = headroom.get(logical_pool) if logical_pool else None
        if isinstance(pool_info, Mapping):
            avail_flag = bool(pool_info.get("available", True))
            avail_pct = pool_info.get("available_pct")
            used_pct = pool_info.get("used_pct")
            rem_quota = pool_info.get("remaining", pool_info.get("remaining_quota"))
            quota_exhausted = (
                not avail_flag
                or (isinstance(avail_pct, (int, float)) and float(avail_pct) <= 0.0)
                or (isinstance(used_pct, (int, float)) and float(used_pct) >= 100.0)
                or (isinstance(rem_quota, (int, float)) and float(rem_quota) <= 0.0)
            )
            if quota_exhausted:
                if strength == ConstraintStrength.HARD:
                    return None, PolicyBlockedError(
                        f"WAIT_RESOURCE: {logical_pool} quota/capacity temporarily exhausted for HARD pinned "
                        f"{provider_key}:{req_eff} (used_pct={used_pct}, available_pct={avail_pct}, "
                        f"source={pool_info.get('source')})",
                        waitable=True,
                        rule_id=rule_id,
                        code="WAIT_RESOURCE",
                        pinned_route=tentative_route.to_dict(),
                    )
                return None, PolicyBlockedError(
                    f"BLOCKED_QUOTA: {logical_pool} 7d quota exhausted "
                    f"(used_pct={used_pct}, source={pool_info.get('source')})",
                    waitable=False,
                    rule_id=rule_id,
                    code="BLOCKED_QUOTA",
                )
            if (
                reserve_pct > 0.0
                and isinstance(used_pct, (int, float))
                and float(used_pct) > (100.0 - reserve_pct)
                and (logical_pool in active_pings or "interactive" in active_pings)
            ):
                return None, PolicyBlockedError(
                    f"WAIT_FOREGROUND_RESERVE: {logical_pool} used_pct={used_pct}% exceeds background "
                    f"reserve ceiling {100.0 - reserve_pct:.1f}% while foreground session is active",
                    waitable=True,
                    rule_id=rule_id,
                    code="WAIT_FOREGROUND_RESERVE",
                    pinned_route=tentative_route.to_dict(),
                )

        return tentative_route, None

    def _parse_explicit_spec(
        self, raw_spec: str, *, role_val: str
    ) -> tuple[str, str, bool]:
        """Parse `explicit_spec` into `(provider_key, effort, had_explicit_effort)`."""
        clean = raw_spec.strip()
        low = clean.lower()
        for banned in self.banned_channels:
            if banned and banned in low:
                raise PolicyBlockedError(
                    f"BLOCKED_BANNED_CHANNEL:{banned}",
                    waitable=False,
                    rule_id="explicit_spec",
                    code="BLOCKED_BANNED_CHANNEL",
                )
        if clean.startswith("cmd:"):
            return clean, "custom", True

        parts = [p.strip() for p in clean.split(":") if p.strip()]
        role_default_eff = "medium" if role_val in REVIEWER_ROLE_VALUES else "high"

        if len(parts) == 1:
            pkey, emb_eff = _resolve_provider_key_from_token(
                parts[0], role_val=role_val, alias_index=self._alias_index
            )
            if pkey is None:
                raise PolicyBlockedError(
                    f"BLOCKED_POLICY: unknown explicit model/pool spec {raw_spec!r}",
                    waitable=False,
                    rule_id="explicit_spec",
                )
            prov_def = str(self.providers[pkey].get("default_effort") or role_default_eff)
            if pkey == "claude-sonnet-5-5":
                eff = "medium"
            elif emb_eff:
                eff = emb_eff
            else:
                eff = prov_def
            return pkey, eff, bool(emb_eff)

        if len(parts) == 2:
            # Distinguish `pool:model` vs `model:effort`
            p1_key, p1_emb = _resolve_provider_key_from_token(
                parts[1], role_val=role_val, alias_index=self._alias_index
            )
            if p1_key is not None and parts[1].lower() not in VALID_EFFORT_TOKENS:
                # `pool:model`
                pool_prefix = parts[0].lower()
                if pool_prefix == "gemini" and self.providers[p1_key]["pool"] != "gemini":
                    raise PolicyBlockedError(
                        f"BLOCKED_BANNED_CHANNEL: non-Gemini model {parts[1]!r} on gemini pool",
                        waitable=False,
                        rule_id="explicit_spec",
                        code="BLOCKED_BANNED_CHANNEL",
                    )
                if p1_emb:
                    return p1_key, p1_emb, True
                eff = "medium" if p1_key == "claude-sonnet-5-5" else str(
                    self.providers[p1_key].get("default_effort") or role_default_eff
                )
                return p1_key, eff, False

            p0_key, _ = _resolve_provider_key_from_token(
                parts[0], role_val=role_val, alias_index=self._alias_index
            )
            if p0_key is not None and parts[1].lower() in VALID_EFFORT_TOKENS:
                return p0_key, parts[1], True

            raise PolicyBlockedError(
                f"BLOCKED_POLICY: unknown explicit provider/model spec {raw_spec!r}",
                waitable=False,
                rule_id="explicit_spec",
            )

        # 3+ parts: `pool:model:effort`
        pool_prefix = parts[0].lower()
        pkey, _ = _resolve_provider_key_from_token(
            parts[1], role_val=role_val, alias_index=self._alias_index
        )
        if pkey is None:
            raise PolicyBlockedError(
                f"BLOCKED_POLICY: unknown model {parts[1]!r} in explicit spec {raw_spec!r}",
                waitable=False,
                rule_id="explicit_spec",
            )
        if pool_prefix == "gemini" and self.providers[pkey]["pool"] != "gemini":
            raise PolicyBlockedError(
                f"BLOCKED_BANNED_CHANNEL: non-Gemini model {parts[1]!r} on gemini pool",
                waitable=False,
                rule_id="explicit_spec",
                code="BLOCKED_BANNED_CHANNEL",
            )
        eff_raw = parts[2]
        if eff_raw.lower() not in VALID_EFFORT_TOKENS:
            raise PolicyBlockedError(
                f"BLOCKED_POLICY: invalid effort {eff_raw!r} in explicit spec {raw_spec!r}",
                waitable=False,
                rule_id="explicit_spec",
            )
        return pkey, eff_raw, True

    @staticmethod
    def _is_nonempty_policy_input(raw_pol: Any, raw_nl: str = "") -> bool:
        if raw_nl and str(raw_nl).strip():
            return True
        if raw_pol is None:
            return False
        if isinstance(raw_pol, str):
            return bool(raw_pol.strip())
        if isinstance(raw_pol, ModelPolicy):
            return bool(
                raw_pol.rules
                or raw_pol.source_nl.strip()
                or raw_pol.review_overrides
                or raw_pol.default_strength == ConstraintStrength.HARD
            )
        if isinstance(raw_pol, Mapping):
            if raw_pol.get("has_active_constraints") is True:
                return True
            for k in (
                "rules",
                "source_nl",
                "routing_policy_nl",
                "review_overrides",
                "disallowed",
                "disallowed_models",
                "disallowed_families",
                "disallowed_pools",
                "allowed_custom_commands",
                "global_policy",
                "direction_policy",
            ):
                val = raw_pol.get(k)
                if val:
                    return True
            dbun = raw_pol.get("disallowed_bundle")
            if isinstance(dbun, Mapping) and any(bool(v) for v in dbun.values()):
                return True
            if raw_pol.get("forced") is True or raw_pol.get("hard") is True or raw_pol.get("allow_fallback") is False:
                return True
            if str(raw_pol.get("default_strength") or "").strip().upper() == ConstraintStrength.HARD.value:
                return True
            return False
        return True

    def _compile_single_scope(
        self,
        raw_pol: Any,
        raw_nl: str,
        *,
        project: str,
        scope_level: int,
        scope_name: str,
    ) -> tuple[dict[str, Any], Mapping[str, Any]]:
        raw_map: Mapping[str, Any] = raw_pol if isinstance(raw_pol, Mapping) else {}
        if isinstance(raw_pol, Mapping) and raw_pol:
            merged_raw = dict(raw_pol)
            if not merged_raw.get("rules") and not merged_raw.get("source_nl") and raw_nl:
                merged_raw["source_nl"] = raw_nl
            compiled = self.compile_routing_policy(
                merged_raw,
                project=project,
                capabilities_path=self.capabilities_path,
            )
        elif raw_pol is not None and not isinstance(raw_pol, Mapping):
            compiled = self.compile_routing_policy(
                raw_pol,
                project=project,
                capabilities_path=self.capabilities_path,
            )
        else:
            compiled = self.compile_routing_policy(
                raw_nl,
                project=project,
                capabilities_path=self.capabilities_path,
            )

        tagged_rules: list[dict[str, Any]] = []
        for idx, r in enumerate(compiled.get("rules") or [], start=1):
            rc = dict(r)
            rc["_scope_level"] = int(scope_level)
            rc["_scope_name"] = scope_name
            if not rc.get("rule_id"):
                rc["rule_id"] = f"{scope_name}_rule_{idx}"
            tagged_rules.append(rc)
        compiled["rules"] = tagged_rules
        return compiled, raw_map

    @staticmethod
    def _read_persisted_global_policy_from_sqlite_path(
        state_db_path: Path,
    ) -> tuple[Any, str]:
        if not state_db_path.is_file():
            return None, ""
        import sqlite3

        conn = sqlite3.connect(f"file:{state_db_path}?mode=ro", uri=True, timeout=5.0)
        try:
            has_tbl = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'global_config'"
            ).fetchone()
            if has_tbl is None:
                return None, ""
            rows = conn.execute(
                "SELECT config_key, value_json FROM global_config WHERE config_key IN ('routing_policy', 'routing_policy_nl')"
            ).fetchall()
            kv = {str(r[0]): json.loads(r[1]) for r in rows if r[1]}
            raw_pol = kv.get("routing_policy")
            raw_nl = str(kv.get("routing_policy_nl") or "").strip()
            return raw_pol, raw_nl
        except Exception:
            return None, ""
        finally:
            conn.close()

    def _load_persisted_global_policy(self, db: Any = None) -> tuple[Any, str]:
        eff_db = db if db is not None else self.db
        if eff_db is not None and hasattr(eff_db, "get_global_config"):
            try:
                raw_pol = eff_db.get_global_config("routing_policy")
                raw_nl = str(eff_db.get_global_config("routing_policy_nl") or "").strip()
                return raw_pol, raw_nl
            except Exception:
                return None, ""

        candidate_home: Path | None = self.arr_home
        if candidate_home is None:
            env_home = os.environ.get("CONJECTRA_HQ_HOME") or os.environ.get("CONJECTRA_HOME")
            if env_home and env_home.strip():
                candidate_home = Path(env_home).expanduser().resolve()
        if candidate_home is not None:
            db_file = (
                candidate_home
                if candidate_home.suffix in (".db", ".sqlite")
                else candidate_home / "state.db"
            )
            return self._read_persisted_global_policy_from_sqlite_path(db_file)
        return None, ""

    def resolve_effective_policy(
        self,
        *,
        goal: ResearchGoal | None = None,
        hyp: HypothesisNode | None = None,
        workflow_id: str | None = None,
        task: Any = None,
        wf_config: Mapping[str, Any] | None = None,
        db: Any = None,
    ) -> dict[str, Any]:
        """Resolve unified effective routing policy across `capabilities-global -> sqlite-global -> direction -> goal -> branch -> workflow -> task`.

        Enforces additive union of `disallowed_*` across all scopes and narrowest-scope-first
        precedence for positive routing rules.
        """
        eff_db = db if db is not None else self.db
        eff_proj = str(getattr(goal, "project", "") or "").strip()
        eff_wf_id = str(workflow_id or getattr(task, "workflow_id", "") or "").strip()
        hparams: Mapping[str, Any] = (
            hyp.params if (hyp is not None and isinstance(getattr(hyp, "params", None), Mapping)) else {}
        )

        eff_wf_cfg: Mapping[str, Any] | None = wf_config if isinstance(wf_config, Mapping) else None
        if eff_wf_cfg is None and eff_wf_id and eff_db is not None and hasattr(eff_db, "get_workflow"):
            try:
                wf_obj = eff_db.get_workflow(eff_wf_id)
                if wf_obj is not None and isinstance(getattr(wf_obj, "config", None), Mapping):
                    eff_wf_cfg = wf_obj.config
            except Exception:
                eff_wf_cfg = None

        scope_entries: list[tuple[int, str, Any, str]] = []

        # 0a. Capabilities-level global scope
        cap_glob_pol = getattr(self, "global_routing_policy", None)
        if self._is_nonempty_policy_input(cap_glob_pol):
            scope_entries.append((-1, "capabilities_global", cap_glob_pol, ""))

        # 0b. Persisted SQLite `global_config["routing_policy"]` (`set_routing_policy(scope="global")`)
        db_glob_pol, db_glob_nl = self._load_persisted_global_policy(eff_db)
        if self._is_nonempty_policy_input(db_glob_pol, db_glob_nl):
            scope_entries.append((0, "global", db_glob_pol, db_glob_nl))

        # 0c. Goal-embedded global_policy fallback (if provided inline on goal)
        if goal is not None:
            inline_glob = getattr(goal, "global_routing_policy", None)
            if inline_glob is None and isinstance(getattr(goal, "routing_policy", None), Mapping):
                inline_glob = goal.routing_policy.get("global_policy")
            if self._is_nonempty_policy_input(inline_glob):
                scope_entries.append((0, "global_inline", inline_glob, ""))

        # 1. Direction scope
        dir_pol = getattr(goal, "direction_routing_policy", None) if goal is not None else None
        if dir_pol is None and goal is not None and isinstance(getattr(goal, "routing_policy", None), Mapping):
            dir_pol = goal.routing_policy.get("direction_policy")
        if dir_pol is None:
            dir_pol = hparams.get("direction_routing_policy")
        if self._is_nonempty_policy_input(dir_pol):
            scope_entries.append((1, "direction", dir_pol, ""))

        # 2. Goal scope
        if goal is not None:
            g_pol = getattr(goal, "routing_policy", None)
            g_nl = str(getattr(goal, "routing_policy_nl", "") or "").strip()
            if self._is_nonempty_policy_input(g_pol, g_nl):
                scope_entries.append((2, "goal", g_pol, g_nl))

        # 3. Hypothesis / branch scope
        if hyp is not None:
            h_attr_pol = getattr(hyp, "routing_policy", None)
            h_attr_nl = str(getattr(hyp, "routing_policy_nl", "") or "").strip()
            if self._is_nonempty_policy_input(h_attr_pol, h_attr_nl):
                scope_entries.append((3, "branch", h_attr_pol, h_attr_nl))
            h_param_pol = hparams.get("routing_policy")
            h_param_nl = str(hparams.get("routing_policy_nl") or "").strip()
            if (h_param_pol is not None or h_param_nl) and h_param_pol is not h_attr_pol:
                if self._is_nonempty_policy_input(h_param_pol, h_param_nl):
                    scope_entries.append((3, "branch", h_param_pol, h_param_nl))

        # 4. Workflow scope (keyed by workflow_id without leaking across sibling workflows)
        wf_pol_candidates: list[Any] = []
        wf_map = hparams.get("workflow_routing_policies")
        if eff_wf_id and isinstance(wf_map, Mapping) and eff_wf_id in wf_map:
            wf_pol_candidates.append(wf_map[eff_wf_id])
        if isinstance(eff_wf_cfg, Mapping) and "routing_policy" in eff_wf_cfg:
            cand_wfp = eff_wf_cfg.get("routing_policy")
            if not wf_pol_candidates or cand_wfp != wf_pol_candidates[-1]:
                wf_pol_candidates.append(cand_wfp)
        for wfp in wf_pol_candidates:
            if self._is_nonempty_policy_input(wfp):
                scope_entries.append((4, "workflow", wfp, ""))

        # 5. Task scope
        if task is not None:
            t_pol = getattr(task, "routing_policy", None)
            if t_pol is None and isinstance(getattr(task, "pinned_route", None), Mapping):
                t_pol = task.pinned_route.get("routing_policy")
            if (
                t_pol is None
                and getattr(task, "resources", None) is not None
                and isinstance(getattr(task.resources, "pinned_route", None), Mapping)
            ):
                t_pol = task.resources.pinned_route.get("routing_policy")
            if self._is_nonempty_policy_input(t_pol):
                scope_entries.append((5, "task", t_pol, ""))

        if not scope_entries:
            empty_pol = ModelPolicy(
                project=eff_proj,
                foreground_reserve_quota_pct=self.foreground_reserve_quota_pct,
            ).to_dict()
            empty_pol["disallowed_bundle"] = _merge_disallowed_bundles()
            empty_pol["allowed_custom_commands"] = _merge_allowed_custom_commands(self.custom_commands)
            empty_pol["has_active_constraints"] = False
            return empty_pol

        compiled_layers: list[tuple[int, str, dict[str, Any], Mapping[str, Any]]] = []
        for s_lvl, s_name, r_pol, r_nl in scope_entries:
            comp_s, raw_m = self._compile_single_scope(
                r_pol,
                r_nl,
                project=eff_proj,
                scope_level=s_lvl,
                scope_name=s_name,
            )
            compiled_layers.append((s_lvl, s_name, comp_s, raw_m))

        merged_rules: list[dict[str, Any]] = []
        review_overrides: dict[str, str] = {}
        default_strength_val = ConstraintStrength.SOFT_FALLBACK.value
        allow_effort_alias = False
        reserve_pct = float(self.foreground_reserve_quota_pct)
        source_nls: list[str] = []

        for _s_lvl, _s_name, comp_s, raw_m in compiled_layers:
            if comp_s.get("source_nl"):
                source_nls.append(str(comp_s["source_nl"]))
            review_overrides.update(dict(comp_s.get("review_overrides") or {}))
            if (
                not raw_m
                or any(k in raw_m for k in ("default_strength", "forced", "hard", "allow_fallback"))
                or str(comp_s.get("default_strength") or "") == ConstraintStrength.HARD.value
            ):
                default_strength_val = str(
                    comp_s.get("default_strength") or ConstraintStrength.SOFT_FALLBACK.value
                )
            if "allow_effort_alias" in raw_m or bool(comp_s.get("allow_effort_alias", False)):
                allow_effort_alias = bool(comp_s.get("allow_effort_alias", False))
            if "foreground_reserve_quota_pct" in raw_m:
                reserve_pct = float(comp_s["foreground_reserve_quota_pct"])
            merged_rules.extend(dict(r) for r in (comp_s.get("rules") or []))

        # Order rules strictly narrowest-scope-first, then specificity rank, then declaration order
        indexed_rules = sorted(
            enumerate(merged_rules),
            key=lambda pair: (
                -int(pair[1].get("_scope_level", 2)),
                _compute_rule_rank(pair[1].get("roles") or [], pair[1].get("tags") or []),
                pair[0],
            ),
        )
        ordered_rules = [r for _, r in indexed_rules]

        merged_disallowed = _merge_disallowed_bundles(
            *(comp_s.get("disallowed_bundle") for _, _, comp_s, _ in compiled_layers)
        )
        merged_custom_cmds = _merge_allowed_custom_commands(
            self.custom_commands,
            *(comp_s.get("allowed_custom_commands") for _, _, comp_s, _ in compiled_layers),
        )
        has_any_disallow = any(bool(v) for v in merged_disallowed.values())
        has_active = any(
            bool(comp_s.get("has_active_constraints", False))
            for _, _, comp_s, _ in compiled_layers
        ) or bool(ordered_rules or has_any_disallow or review_overrides)

        return {
            "source_nl": "; ".join(source_nls),
            "project": eff_proj,
            "rules": ordered_rules,
            "review_overrides": review_overrides,
            "default_strength": default_strength_val,
            "allow_effort_alias": allow_effort_alias,
            "foreground_reserve_quota_pct": reserve_pct,
            "disallowed_bundle": merged_disallowed,
            "allowed_custom_commands": merged_custom_cmds,
            "has_active_constraints": has_active,
        }

    def has_effective_policy(
        self,
        *,
        goal: ResearchGoal | None = None,
        hyp: HypothesisNode | None = None,
        workflow_id: str | None = None,
        task: Any = None,
        wf_config: Mapping[str, Any] | None = None,
        db: Any = None,
    ) -> bool:
        """Return True if any scope (`global..task`) defines active routing constraints."""
        eff = self.resolve_effective_policy(
            goal=goal,
            hyp=hyp,
            workflow_id=workflow_id,
            task=task,
            wf_config=wf_config,
            db=db,
        )
        return bool(eff.get("has_active_constraints", False))

    def matching_policy_rules(
        self,
        role: ScientificRole | str,
        *,
        goal: ResearchGoal | None = None,
        hyp: HypothesisNode | None = None,
        workflow_id: str | None = None,
        task: Any = None,
        wf_config: Mapping[str, Any] | None = None,
        db: Any = None,
    ) -> tuple[dict[str, Any], list[Mapping[str, Any]], set[str]]:
        """Return `(effective_policy, matching_rules_in_precedence_order, hyp_tags)` for `(role, scope)`."""
        role_val = normalize_role(role)
        compiled_policy = self.resolve_effective_policy(
            goal=goal,
            hyp=hyp,
            workflow_id=workflow_id,
            task=task,
            wf_config=wf_config,
            db=db,
        )
        hyp_tags: set[str] = set()
        if hyp is not None:
            for tg in getattr(hyp, "tags", ()) or ():
                if str(tg).strip():
                    hyp_tags.add(str(tg).strip().lower())
            extra_t = (getattr(hyp, "params", None) or {}).get("tags")
            if isinstance(extra_t, list):
                for tg in extra_t:
                    if str(tg).strip():
                        hyp_tags.add(str(tg).strip().lower())

        goal_proj = (goal.project if goal is not None else "").strip().lower()
        rules = list(compiled_policy.get("rules") or [])
        rules_sorted = sorted(
            enumerate(rules),
            key=lambda pair: (
                -int(pair[1].get("_scope_level", 2)),
                _compute_rule_rank(pair[1].get("roles") or [], pair[1].get("tags") or []),
                pair[0],
            ),
        )
        matching_rules: list[Mapping[str, Any]] = []
        for _, rule in rules_sorted:
            proj_scope = str(rule.get("project_scope") or "").strip().lower()
            if proj_scope and goal_proj and proj_scope != goal_proj:
                continue
            if bool(rule.get("author_only", False)) and role_val in REVIEWER_ROLE_VALUES:
                continue
            r_roles = {normalize_role(str(x)) for x in (rule.get("roles") or []) if str(x).strip()}
            r_tags = {str(x).strip().lower() for x in (rule.get("tags") or []) if str(x).strip()}
            if r_roles and role_val not in r_roles:
                continue
            if r_tags and not (r_tags & hyp_tags):
                continue
            matching_rules.append(rule)
        return compiled_policy, matching_rules, hyp_tags

    def has_matching_rule(
        self,
        role: ScientificRole | str,
        *,
        goal: ResearchGoal | None = None,
        hyp: HypothesisNode | None = None,
        workflow_id: str | None = None,
        task: Any = None,
        wf_config: Mapping[str, Any] | None = None,
        db: Any = None,
    ) -> bool:
        _, matching_rules, _ = self.matching_policy_rules(
            role,
            goal=goal,
            hyp=hyp,
            workflow_id=workflow_id,
            task=task,
            wf_config=wf_config,
            db=db,
        )
        return bool(matching_rules)

    def has_matching_hard_rule(
        self,
        role: ScientificRole | str,
        *,
        goal: ResearchGoal | None = None,
        hyp: HypothesisNode | None = None,
        workflow_id: str | None = None,
        task: Any = None,
        wf_config: Mapping[str, Any] | None = None,
        db: Any = None,
    ) -> bool:
        _, matching_rules, _ = self.matching_policy_rules(
            role,
            goal=goal,
            hyp=hyp,
            workflow_id=workflow_id,
            task=task,
            wf_config=wf_config,
            db=db,
        )
        return any(
            str(r.get("strength", "")).upper() == ConstraintStrength.HARD.value
            for r in matching_rules
        )

    def revalidate_pinned_route(
        self,
        role: ScientificRole | str,
        pinned_route: ResolvedModelRoute | Mapping[str, Any],
        *,
        goal: ResearchGoal | None = None,
        hyp: HypothesisNode | None = None,
        workflow_id: str | None = None,
        task: Any = None,
        wf_config: Mapping[str, Any] | None = None,
        db: Any = None,
        avoid_families: Sequence[str] = (),
        exclude_specs: Sequence[str] = (),
        execution_host: str | None = None,
        force_multipool: bool = False,
    ) -> ResolvedModelRoute:
        """Revalidate a previously pinned route against effective scoped HARD rules, disallows, and live quota."""
        role_val = normalize_role(role)
        r_obj = (
            pinned_route
            if isinstance(pinned_route, ResolvedModelRoute)
            else ResolvedModelRoute.from_dict(pinned_route)
        )
        compiled_policy, matching_rules, _ = self.matching_policy_rules(
            role_val,
            goal=goal,
            hyp=hyp,
            workflow_id=workflow_id,
            task=task,
            wf_config=wf_config,
            db=db,
        )
        top_hard = next(
            (
                r
                for r in matching_rules
                if str(r.get("strength", "")).upper() == ConstraintStrength.HARD.value
            ),
            None,
        )
        if top_hard is not None:
            allow_alias = bool(
                top_hard.get("allow_effort_alias", compiled_policy.get("allow_effort_alias", False))
            )
            r_pkey, _ = _resolve_provider_key_from_token(
                r_obj.canonical_model,
                role_val=role_val,
                alias_index=self._alias_index,
            )
            cand_ok = False
            for cand in top_hard.get("candidates") or []:
                c_pk = str(cand.get("provider_key") or "")
                c_eff = str(cand.get("effort") or "").strip().lower()
                if r_obj.pool == "cmd":
                    if c_pk.lower() == f"cmd:{r_obj.canonical_model}".lower():
                        cand_ok = True
                        break
                    continue
                if r_pkey is not None and c_pk == r_pkey:
                    req_l = r_obj.requested_effort.strip().lower()
                    wire_l = r_obj.wire_effort.strip().lower()
                    if c_eff in (req_l, wire_l) or allow_alias:
                        cand_ok = True
                        break
            if not cand_ok:
                raise PolicyBlockedError(
                    f"BLOCKED_POLICY: pinned_route {r_obj.canonical_spec()} violates active HARD rule "
                    f"{top_hard.get('rule_id')!r}",
                    waitable=False,
                    rule_id=str(top_hard.get("rule_id") or "hard_rule"),
                    code="BLOCKED_POLICY",
                )

        exp_spec = (
            f"cmd:{r_obj.canonical_model}"
            if r_obj.pool == "cmd"
            else f"{r_obj.pool}:{r_obj.canonical_model}:{r_obj.requested_effort}"
        )
        return self.resolve_route(
            role_val,
            goal=goal,
            hyp=hyp,
            workflow_id=workflow_id,
            task=task,
            wf_config=wf_config,
            db=db,
            explicit_spec=exp_spec,
            avoid_families=avoid_families,
            exclude_specs=exclude_specs,
            execution_host=execution_host or r_obj.execution_host,
            strength=ConstraintStrength.HARD,
            force_multipool=force_multipool,
        )

    def resolve_route(
        self,
        role: ScientificRole | str,
        *,
        goal: ResearchGoal | None = None,
        hyp: HypothesisNode | None = None,
        workflow_id: str | None = None,
        task: Any = None,
        wf_config: Mapping[str, Any] | None = None,
        db: Any = None,
        explicit_spec: str | None = None,
        avoid_families: Sequence[str] = (),
        prefer_avoid_families: Sequence[str] = (),
        exclude_specs: Sequence[str] = (),
        execution_host: str | None = None,
        strength: ConstraintStrength | str | None = None,
        force_multipool: bool = False,
    ) -> ResolvedModelRoute:
        """Resolve and validate one immutable ResolvedModelRoute against capabilities & live headroom."""
        role_val = normalize_role(role)
        host = (execution_host or default_execution_host()).strip().lower()
        avoid_set = {str(f).strip().lower() for f in avoid_families if str(f).strip()}
        prefer_avoid_set = avoid_set | {
            str(f).strip().lower() for f in prefer_avoid_families if str(f).strip()
        }
        excluded_set = {str(s).strip().lower() for s in exclude_specs if str(s).strip()}

        headroom = self.arbiter.read_pool_headroom() if self.arbiter is not None else {}
        active_pings: dict[str, dict[str, Any]] = {}
        if self.arr_home is not None:
            from conjectra.arbiter import read_active_foreground_pings

            active_pings = read_active_foreground_pings(self.arr_home)

        compiled_policy, matching_rules, hyp_tags = self.matching_policy_rules(
            role_val,
            goal=goal,
            hyp=hyp,
            workflow_id=workflow_id,
            task=task,
            wf_config=wf_config,
            db=db,
        )
        has_active_policy = bool(
            compiled_policy.get("has_active_constraints", False)
            or goal is not None
            or hyp is not None
            or bool(workflow_id)
            or bool(wf_config)
            or task is not None
        )
        allowed_custom_cmds = dict(compiled_policy.get("allowed_custom_commands") or self.custom_commands)

        reserve_pct = float(
            compiled_policy.get("foreground_reserve_quota_pct", self.foreground_reserve_quota_pct)
        )
        policy_allow_alias = bool(compiled_policy.get("allow_effort_alias", False))
        policy_disallowed = _merge_disallowed_bundles(compiled_policy.get("disallowed_bundle"))

        effective_disallowed = _merge_disallowed_bundles(
            policy_disallowed,
            *(r.get("disallowed_bundle") for r in matching_rules),
        )

        # Rank 1: Explicit spec on call
        if explicit_spec is not None and explicit_spec.strip() and explicit_spec.strip().lower() != "auto":
            pkey, req_eff, had_explicit_eff = self._parse_explicit_spec(
                explicit_spec, role_val=role_val
            )
            if strength is not None:
                eff_strength = (
                    strength
                    if isinstance(strength, ConstraintStrength)
                    else ConstraintStrength(str(strength))
                )
            elif had_explicit_eff:
                eff_strength = ConstraintStrength.HARD
            else:
                eff_strength = ConstraintStrength(
                    str(compiled_policy.get("default_strength", ConstraintStrength.SOFT_FALLBACK.value))
                )
            route, err = self._evaluate_candidate(
                provider_key=pkey,
                requested_effort=req_eff,
                role_val=role_val,
                host=host,
                strength=eff_strength,
                allow_effort_alias=policy_allow_alias or (eff_strength == ConstraintStrength.SOFT_FALLBACK),
                rule_id="explicit_spec",
                avoid_set=avoid_set,
                excluded_set=excluded_set,
                headroom=headroom,
                active_pings=active_pings,
                reserve_pct=reserve_pct,
                force_multipool=force_multipool,
                disallowed_bundle=effective_disallowed,
                has_active_policy=has_active_policy,
                allowed_custom_commands=allowed_custom_cmds,
            )
            if route is not None:
                return route
            assert err is not None
            raise err

        # Ranks 2..5: Matching compiled policy rules in strict scope + specificity order
        last_soft_err: PolicyBlockedError | None = None
        for rule in matching_rules:
            r_strength = ConstraintStrength(
                str(rule.get("strength", ConstraintStrength.SOFT_FALLBACK.value))
            )
            r_allow_alias = bool(rule.get("allow_effort_alias", policy_allow_alias))
            r_id = str(rule.get("rule_id") or "policy_rule")
            r_cands = list(rule.get("candidates") or [])

            # Try strict prefer_avoid_set first when distinct from avoid_set, then avoid_set
            family_passes = (
                [prefer_avoid_set, avoid_set]
                if prefer_avoid_set != avoid_set
                else [avoid_set]
            )
            rule_errors: list[PolicyBlockedError] = []
            for active_avoid in family_passes:
                for cand in r_cands:
                    route, err = self._evaluate_candidate(
                        provider_key=str(cand["provider_key"]),
                        requested_effort=str(cand["effort"]),
                        role_val=role_val,
                        host=host,
                        strength=r_strength,
                        allow_effort_alias=r_allow_alias,
                        rule_id=r_id,
                        avoid_set=active_avoid,
                        excluded_set=excluded_set,
                        headroom=headroom,
                        active_pings=active_pings,
                        reserve_pct=reserve_pct,
                        force_multipool=force_multipool,
                        disallowed_bundle=effective_disallowed,
                        has_active_policy=has_active_policy,
                        allowed_custom_commands=allowed_custom_cmds,
                        candidate_family=str(cand.get("family") or ""),
                    )
                    if route is not None:
                        return route
                    if err is not None and active_avoid is avoid_set:
                        rule_errors.append(err)

            if r_strength == ConstraintStrength.HARD:
                if rule_errors:
                    waitable_err = next((e for e in rule_errors if e.waitable), None)
                    raise (waitable_err or rule_errors[0])
                raise PolicyBlockedError(
                    f"BLOCKED_POLICY: HARD rule {r_id!r} matched role={role_val} tags={sorted(hyp_tags)} "
                    "but zero candidates succeeded",
                    waitable=False,
                    rule_id=r_id,
                )
            if rule_errors:
                last_soft_err = rule_errors[0]

        # Built-in role defaults (SOFT_FALLBACK) — strictly honoring effective_disallowed
        default_cands = self._default_candidates_for_role(role_val)
        family_passes = (
            [prefer_avoid_set, avoid_set]
            if prefer_avoid_set != avoid_set
            else [avoid_set]
        )
        for active_avoid in family_passes:
            for cand in default_cands:
                route, err = self._evaluate_candidate(
                    provider_key=cand["provider_key"],
                    requested_effort=cand["effort"],
                    role_val=role_val,
                    host=host,
                    strength=ConstraintStrength.SOFT_FALLBACK,
                    allow_effort_alias=True,
                    rule_id="role_default",
                    avoid_set=active_avoid,
                    excluded_set=excluded_set,
                    headroom=headroom,
                    active_pings=active_pings,
                    reserve_pct=reserve_pct,
                    force_multipool=force_multipool,
                    disallowed_bundle=effective_disallowed,
                    has_active_policy=has_active_policy,
                    allowed_custom_commands=allowed_custom_cmds,
                )
                if route is not None:
                    return route
                if err is not None and active_avoid is avoid_set:
                    last_soft_err = err

        if last_soft_err is not None:
            raise last_soft_err
        raise PolicyBlockedError(
            f"BLOCKED_QUOTA: no healthy cross-family route available for role={role_val} "
            f"(avoid_families={sorted(avoid_set)}, exclude_specs={sorted(excluded_set)})",
            waitable=False,
            rule_id="role_default",
            code="BLOCKED_QUOTA",
        )


__all__ = [
    "DEFAULT_CAPABILITIES_PATH",
    "ModelRouter",
    "PolicyBlockedError",
    "normalize_role",
]
