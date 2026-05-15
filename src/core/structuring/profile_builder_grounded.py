from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from .grounded_prompt import build_grounded_extraction_prompt, build_system_message
from .grounded_reporting import generate_reports, write_json
from .grounding_validator import validate_and_ground
from .markdown_normalizer import normalize_markdown

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

LOGGER = logging.getLogger("profile_builder_grounded")

GROQ_BASE_URL = os.getenv("PROFILE_BUILDER_SECONDARY_GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.getenv("PROFILE_BUILDER_SECONDARY_GROQ_MODEL", os.getenv("PROFILE_BUILDER_MODEL", "llama-3.3-70b-versatile"))
OLLAMA_BASE_URL = os.getenv("PROFILE_BUILDER_LOCAL_OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("PROFILE_BUILDER_LOCAL_OLLAMA_MODEL", "llama3.2:3b")
PROVIDER_SATURATION_CONSECUTIVE = int(os.getenv("PROFILE_BUILDER_GROUNDED_PROVIDER_SATURATION_CONSECUTIVE", "3"))


def configure_logging() -> None:
    if LOGGER.handlers:
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_accepted_entries(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [
        row
        for row in rows
        if row.get("eligible_for_module2") is True
        and row.get("handoff_lane") == "accepted"
        and row.get("artifact_path")
    ]


def slug_for_entry(entry: dict[str, Any], artifact_path: Path | None = None) -> str:
    path = artifact_path or Path(entry["artifact_path"])
    source_format = entry.get("source_format") or "unknown"
    return f"{source_format}_{path.stem}"


def call_provider(
    provider: str,
    system_message: str,
    user_prompt: str,
    *,
    provider_timeout: float,
    ollama_timeout: float,
) -> tuple[dict[str, Any], str, str]:
    if provider == "groq_secondary":
        api_key = (
            os.getenv("PROFILE_BUILDER_SECONDARY_GROQ_API_KEY")
            or os.getenv("GROQ_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not api_key:
            raise RuntimeError("groq_secondary_not_configured")
        return _call_openai_compatible(
            base_url=GROQ_BASE_URL,
            api_key=api_key,
            model=GROQ_MODEL,
            provider_route=provider,
            system_message=system_message,
            user_prompt=user_prompt,
            timeout_seconds=provider_timeout,
        )
    if provider == "ollama_local":
        return _call_openai_compatible(
            base_url=OLLAMA_BASE_URL,
            api_key=os.getenv("PROFILE_BUILDER_LOCAL_OLLAMA_API_KEY") or "ollama",
            model=OLLAMA_MODEL,
            provider_route=provider,
            system_message=system_message,
            user_prompt=user_prompt,
            timeout_seconds=ollama_timeout,
        )
    raise RuntimeError(f"unsupported_provider:{provider}")


def _call_openai_compatible(
    *,
    base_url: str,
    api_key: str,
    model: str,
    provider_route: str,
    system_message: str,
    user_prompt: str,
    timeout_seconds: float,
) -> tuple[dict[str, Any], str, str]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    return parse_json_response(content), model, provider_route


def provider_preflight(provider: str, *, ollama_timeout_seconds: float) -> dict[str, Any]:
    if provider == "groq_secondary":
        api_key = (
            os.getenv("PROFILE_BUILDER_SECONDARY_GROQ_API_KEY")
            or os.getenv("GROQ_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        key_present = bool(api_key)
        return {
            "provider": provider,
            "available": key_present,
            "status": "ok" if key_present else "groq_missing_api_key",
            "action_required": None if key_present else "set GROQ_API_KEY environment variable",
            "groq_key_present": key_present,
            "ollama_model_detected": None,
        }
    if provider == "ollama_local":
        model_names: list[str] = []
        try:
            response = requests.get(OLLAMA_BASE_URL.rstrip("/") + "/models", timeout=5)
            if response.status_code < 400:
                data = response.json()
                for item in data.get("data") or []:
                    model_id = item.get("id")
                    if isinstance(model_id, str) and model_id.strip():
                        model_names.append(model_id.strip())
            else:
                return {
                    "provider": provider,
                    "available": False,
                    "status": f"ollama_models_endpoint_status_{response.status_code}",
                    "action_required": "start Ollama server or choose Groq",
                    "groq_key_present": None,
                    "ollama_model_detected": None,
                }
        except Exception as exc:
            return {
                "provider": provider,
                "available": False,
                "status": "ollama_unreachable",
                "action_required": f"start Ollama server or choose Groq ({exc})",
                "groq_key_present": None,
                "ollama_model_detected": None,
            }
        model_detected = OLLAMA_MODEL in model_names if model_names else False
        if not model_detected:
            return {
                "provider": provider,
                "available": False,
                "status": "ollama_model_missing",
                "action_required": f"pull or configure Ollama model {OLLAMA_MODEL}",
                "groq_key_present": None,
                "ollama_model_detected": None,
                "ollama_models": model_names,
            }
        return {
            "provider": provider,
            "available": True,
            "status": "ok",
            "action_required": None,
            "groq_key_present": None,
            "ollama_model_detected": OLLAMA_MODEL,
            "ollama_models": model_names,
            "ollama_timeout": ollama_timeout_seconds,
        }
    return {
        "provider": provider,
        "available": False,
        "status": f"unsupported_provider:{provider}",
        "action_required": "choose a supported provider",
        "groq_key_present": None,
        "ollama_model_detected": None,
    }


def build_provider_diagnostic(selected_provider: str, fallback_provider: str, *, ollama_timeout_seconds: float) -> dict[str, Any]:
    primary = provider_preflight(selected_provider, ollama_timeout_seconds=ollama_timeout_seconds)
    fallback = provider_preflight(fallback_provider, ollama_timeout_seconds=ollama_timeout_seconds)
    groq_info = primary if primary["provider"] == "groq_secondary" else fallback if fallback["provider"] == "groq_secondary" else {}
    ollama_info = primary if primary["provider"] == "ollama_local" else fallback if fallback["provider"] == "ollama_local" else {}

    if primary.get("available"):
        preflight_status = "ok"
        recommended = selected_provider
        action = None
    elif fallback.get("available"):
        preflight_status = "degraded_fallback_only"
        recommended = fallback_provider
        action = f"primary provider unavailable ({primary.get('status')}); fallback can be used"
    else:
        preflight_status = "provider_preflight_failed"
        recommended = None
        action = fallback.get("action_required") or primary.get("action_required") or "configure at least one provider"

    return {
        "generated_at": utc_now(),
        "groq_key_present": bool(groq_info.get("groq_key_present")),
        "groq_available": bool(groq_info.get("available")) if groq_info else False,
        "groq_status": groq_info.get("status"),
        "ollama_available": bool(ollama_info.get("available")) if ollama_info else False,
        "ollama_status": ollama_info.get("status"),
        "ollama_model_detected": ollama_info.get("ollama_model_detected"),
        "ollama_models": ollama_info.get("ollama_models", []),
        "ollama_timeout": ollama_timeout_seconds,
        "selected_provider": selected_provider,
        "fallback_provider": fallback_provider,
        "provider_preflight_status": preflight_status,
        "recommended_provider": recommended,
        "recommended_next_action": action,
        "providers": {
            selected_provider: primary,
            fallback_provider: fallback,
        },
    }


def parse_json_response(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def classify_provider_error(provider: str, error_text: str) -> tuple[str, str]:
    lower = error_text.lower()
    if "429" in lower or "too many requests" in lower:
        return "provider_rate_limited", "wait for Groq rate limit window or reduce request rate"
    if "timed out" in lower or "timeout" in lower:
        if provider == "ollama_local":
            return "ollama_timeout", "start Ollama server or choose Groq"
        return "provider_timeout", "increase provider timeout or retry later"
    if "not_configured" in lower or "missing_api_key" in lower:
        return "groq_missing_api_key", "set GROQ_API_KEY environment variable"
    return "provider_error", "inspect provider logs"


def call_provider_with_retry(
    *,
    provider: str,
    fallback_provider: str,
    system_message: str,
    user_prompt: str,
    provider_timeout: float,
    ollama_timeout: float,
    max_retries: int,
    rate_limit_sleep: float,
    backoff_multiplier: float,
) -> tuple[dict[str, Any], str, str, list[str], str | None, str | None]:
    provider_errors: list[str] = []
    retries_used = 0
    while True:
        try:
            llm_output, model_used, provider_used = call_provider(
                provider,
                system_message,
                user_prompt,
                provider_timeout=provider_timeout,
                ollama_timeout=ollama_timeout,
            )
            return llm_output, model_used, provider_used, provider_errors, None, None
        except Exception as exc:
            error_text = str(exc)
            provider_errors.append(f"{provider}:{error_text}")
            status, action_required = classify_provider_error(provider, error_text)
            is_rate_limited = status == "provider_rate_limited"
            if is_rate_limited and retries_used < max_retries:
                sleep_seconds = rate_limit_sleep * (backoff_multiplier ** retries_used)
                retries_used += 1
                LOGGER.warning(
                    "Primary grounded provider rate limited; retrying | provider=%s | retry=%s/%s | sleep=%.1fs",
                    provider,
                    retries_used,
                    max_retries,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)
                continue
            LOGGER.warning("Primary grounded provider failed | provider=%s | error=%s", provider, error_text)
            break

    try:
        llm_output, model_used, provider_used = call_provider(
            fallback_provider,
            system_message,
            user_prompt,
            provider_timeout=provider_timeout,
            ollama_timeout=ollama_timeout,
        )
        return llm_output, model_used, provider_used, provider_errors, None, None
    except Exception as fallback_exc:
        fallback_error = str(fallback_exc)
        provider_errors.append(f"{fallback_provider}:{fallback_error}")
        status, action_required = classify_provider_error(fallback_provider, fallback_error)
        return {}, "", provider, provider_errors, status, action_required


def collect_output_items(output_dir: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for profile_path in sorted((output_dir / "profiles" / "grounded_profiles").glob("*.json")):
        try:
            grounded_profile = json.loads(profile_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        slug = profile_path.stem
        legacy_path = output_dir / "profiles" / "legacy_projection" / f"{slug}.json"
        log_path = output_dir / "logs" / f"{slug}.json"
        normalization = grounded_profile.get("normalization") or {}
        grounding = grounded_profile.get("grounding") or {}
        results.append(
            {
                "source_path": grounded_profile.get("source_path"),
                "artifact_path": grounded_profile.get("artifact_path"),
                "profile_file": str(profile_path),
                "legacy_projection_file": str(legacy_path),
                "log_file": str(log_path),
                "status": grounded_profile.get("status", "success"),
                "profile_kind": grounded_profile.get("profile_kind"),
                "provider_used": grounded_profile.get("provider_used"),
                "model_used": grounded_profile.get("model_used"),
                "reliability_score": grounding.get("reliability_score", 0.0),
                "hallucination_risk": grounding.get("hallucination_risk"),
                "quality_flags": grounding.get("quality_flags") or [],
                "normalization_quality_flags": normalization.get("quality_flags") or [],
                "fields_nullified": grounding.get("fields_nullified") or [],
                "fields_supported": grounding.get("fields_supported") or [],
                "fields_unsupported": grounding.get("fields_unsupported") or [],
                "detected_templates": normalization.get("detected_templates") or [],
            }
        )
    return results


def build_existing_artifact_map(output_dir: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for profile_path in (output_dir / "profiles" / "grounded_profiles").glob("*.json"):
        try:
            payload = json.loads(profile_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        artifact_path = payload.get("artifact_path")
        if isinstance(artifact_path, str) and artifact_path:
            mapping[artifact_path] = profile_path
    return mapping


def write_resume_checkpoint(output_dir: Path, checkpoint: dict[str, Any]) -> None:
    checkpoint["generated_at_utc"] = utc_now()
    write_json(output_dir / "reports" / "resume_checkpoint.json", checkpoint)


def process_entry(
    entry: dict[str, Any],
    *,
    output_dir: Path,
    provider: str,
    fallback_provider: str,
    provider_timeout: float,
    ollama_timeout: float,
    max_retries: int,
    rate_limit_sleep: float,
    backoff_multiplier: float,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    artifact_path = Path(entry["artifact_path"])
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    markdown = artifact.get("markdown") or ""
    raw_text = artifact.get("raw_text") or ""
    document_confidence_score = float(
        entry.get("document_confidence_score")
        or (artifact.get("document_confidence") or {}).get("score")
        or 0.0
    )
    normalized = normalize_markdown(markdown, raw_text)
    prompt = build_grounded_extraction_prompt(
        normalized["cleaned_markdown"],
        normalized["sections_found"],
        normalized["detected_templates"],
        document_confidence_score,
        normalized["header_info"],
    )
    llm_output, model_used, provider_used, provider_errors, provider_status, action_required = call_provider_with_retry(
        provider=provider,
        fallback_provider=fallback_provider,
        system_message=build_system_message(),
        user_prompt=prompt,
        provider_timeout=provider_timeout,
        ollama_timeout=ollama_timeout,
        max_retries=max_retries,
        rate_limit_sleep=rate_limit_sleep,
        backoff_multiplier=backoff_multiplier,
    )
    if provider_status is not None:
        return None, {
            "source_path": entry.get("source_path"),
            "artifact_path": str(artifact_path),
            "provider_used": provider_used,
            "failure_type": "provider_error",
            "provider_status": provider_status,
            "action_required": action_required,
            "error": " | ".join(provider_errors),
        }

    grounded = validate_and_ground(
        llm_output,
        normalized["cleaned_markdown"],
        normalized["detected_templates"],
        document_confidence_score,
    )
    source_format = entry.get("source_format") or artifact.get("source_format") or "unknown"
    slug = f"{source_format}_{artifact_path.stem}"
    grounded_path = output_dir / "profiles" / "grounded_profiles" / f"{slug}.json"
    legacy_path = output_dir / "profiles" / "legacy_projection" / f"{slug}.json"
    log_path = output_dir / "logs" / f"{slug}.json"

    grounded_profile = {
        "source_path": entry.get("source_path") or artifact.get("source_path"),
        "artifact_path": str(artifact_path),
        "source_format": source_format,
        "status": "success",
        "run_id": "module2_v2_grounded_all",
        "profile_kind": grounded["profile_kind"],
        "provider_used": provider_used,
        "model_used": model_used,
        "document_confidence_score": document_confidence_score,
        "normalization": normalized,
        "grounding": {
            "reliability_score": grounded["reliability_score"],
            "hallucination_risk": grounded["hallucination_risk"],
            "quality_flags": grounded["quality_flags"],
            "fields_nullified": grounded["fields_nullified"],
            "fields_supported": grounded["fields_supported"],
            "fields_unsupported": grounded["fields_unsupported"],
        },
        "profile": grounded["grounded_profile"],
        "metadata": {
            "created_at": utc_now(),
            "pipeline": "module2_v2_grounded",
            "old_module2_untouched": True,
        },
    }
    legacy_projection = build_legacy_projection(grounded_profile)

    write_json(grounded_path, grounded_profile)
    write_json(legacy_path, legacy_projection)
    write_json(log_path, {
        "source_path": grounded_profile["source_path"],
        "artifact_path": grounded_profile["artifact_path"],
        "provider_errors": provider_errors,
        "prompt_length": len(prompt),
        "llm_status": llm_output.get("status"),
        "fields_nullified": grounded["fields_nullified"],
        "quality_flags": grounded["quality_flags"],
    })

    item = {
        "source_path": grounded_profile["source_path"],
        "artifact_path": grounded_profile["artifact_path"],
        "profile_file": str(grounded_path),
        "legacy_projection_file": str(legacy_path),
        "log_file": str(log_path),
        "status": "success",
        "profile_kind": grounded["profile_kind"],
        "provider_used": provider_used,
        "model_used": model_used,
        "reliability_score": grounded["reliability_score"],
        "hallucination_risk": grounded["hallucination_risk"],
        "quality_flags": grounded["quality_flags"],
        "normalization_quality_flags": normalized["quality_flags"],
        "fields_nullified": grounded["fields_nullified"],
        "fields_supported": grounded["fields_supported"],
        "fields_unsupported": grounded["fields_unsupported"],
        "detected_templates": normalized["detected_templates"],
    }
    return item, None


def build_legacy_projection(grounded_profile: dict[str, Any]) -> dict[str, Any]:
    profile = grounded_profile["profile"]
    bio = profile.get("bio") or {}
    expertise = profile.get("expertise") or {}
    legacy_education = []
    for item in profile.get("education") or []:
        legacy_education.append({
            "degree": item.get("degree"),
            "school": item.get("institution"),
            "year": str(item.get("year")) if item.get("year") is not None else None,
        })
    legacy_experiences = []
    for item in profile.get("experiences") or []:
        legacy_experiences.append({
            "job_title": item.get("job_title"),
            "company": item.get("company"),
            "start_date": item.get("start_date"),
            "end_date": item.get("end_date"),
            "city": item.get("city"),
            "responsibilities": item.get("responsibilities") or [],
        })
    return {
        "source_path": grounded_profile["source_path"],
        "artifact_path": grounded_profile["artifact_path"],
        "source_format": grounded_profile["source_format"],
        "status": grounded_profile["status"],
        "mode": "grounded-v2-dry-output",
        "error": None,
        "run_id": grounded_profile["run_id"],
        "profile": {
            "source_id": grounded_profile["artifact_path"],
            "profile_kind": grounded_profile["profile_kind"],
            "bio": {
                "full_name": bio.get("full_name"),
                "email": bio.get("email"),
                "phone": bio.get("phone"),
                "location": bio.get("location"),
            },
            "expertise": {
                "summary": expertise.get("summary") or "",
                "hard_skills": expertise.get("hard_skills") or [],
                "soft_skills": expertise.get("soft_skills") or [],
            },
            "experiences": legacy_experiences,
            "education": legacy_education,
            "metadata": {
                "extraction_date": utc_now(),
                "model_used": grounded_profile["model_used"],
                "provider_route": grounded_profile["provider_used"],
                "confidence_score": grounded_profile["grounding"]["reliability_score"],
                "grounded_v2": True,
                "hallucination_risk": grounded_profile["grounding"]["hallucination_risk"],
                "quality_flags": grounded_profile["grounding"]["quality_flags"],
            },
        },
    }


def ensure_output_dirs(output_dir: Path) -> None:
    for path in (
        output_dir / "profiles" / "grounded_profiles",
        output_dir / "profiles" / "legacy_projection",
        output_dir / "reports",
        output_dir / "logs",
    ):
        path.mkdir(parents=True, exist_ok=True)


def write_provider_diagnostic_report(output_dir: Path, diagnostic: dict[str, Any]) -> None:
    write_json(output_dir / "reports" / "provider_diagnostic_report.json", diagnostic)


def print_provider_diagnostic(diagnostic: dict[str, Any]) -> None:
    print(f"GROQ_API_KEY found: {str(bool(diagnostic.get('groq_key_present'))).lower()}")
    print(f"Groq provider available: {str(bool(diagnostic.get('groq_available'))).lower()}")
    print(f"Ollama reachable: {str(bool(diagnostic.get('ollama_available'))).lower()}")
    print(
        "Ollama model detected: "
        + (str(diagnostic.get("ollama_model_detected")) if diagnostic.get("ollama_model_detected") else "missing")
    )
    print(f"Recommended provider to use: {diagnostic.get('recommended_provider')}")
    print(f"Provider preflight status: {diagnostic.get('provider_preflight_status')}")
    if diagnostic.get("recommended_next_action"):
        print(f"Clear error message: {diagnostic.get('recommended_next_action')}")


def write_preflight_failed_reports(
    *,
    output_dir: Path,
    accepted_count: int,
    diagnostic: dict[str, Any],
    compare_with: Path | None,
) -> dict[str, Any]:
    run_report = {
        "generated_at": utc_now(),
        "accepted_count": accepted_count,
        "processed": 0,
        "success": 0,
        "failed": 0,
        "provider_preflight_status": diagnostic["provider_preflight_status"],
        "provider_status": {
            "selected_provider": diagnostic.get("providers", {}).get(diagnostic["selected_provider"], {}).get("status"),
            "fallback_provider": diagnostic.get("providers", {}).get(diagnostic["fallback_provider"], {}).get("status"),
        },
        "action_required": diagnostic.get("recommended_next_action"),
        "items": [],
        "failures": [],
    }
    quality_report = {
        "generated_at": utc_now(),
        "total_accepted_cvs_found": accepted_count,
        "total_processed": 0,
        "complete_profile": 0,
        "partial_profile": 0,
        "minimal_profile": 0,
        "unreadable": 0,
        "failed": 0,
        "average_reliability_score": 0.0,
        "hallucination_risk_distribution": {},
        "templates_detected": {},
        "fields_nullified_count": 0,
        "unsupported_field_types": {},
        "mongodb_import_readiness": "not_ready_provider_integration_needed",
        "faiss_rebuild_readiness": "not_ready_provider_integration_needed",
        "provider_preflight_status": diagnostic["provider_preflight_status"],
        "provider_status": diagnostic.get("providers"),
        "action_required": diagnostic.get("recommended_next_action"),
    }
    reduction_report = {
        "generated_at": utc_now(),
        "comparison_with_old_module2": {
            "available": bool(compare_with and compare_with.exists()),
            "old_folder": str(compare_with) if compare_with else None,
            "matched_profiles": 0,
            "new_profiles_compared": 0,
            "examples_before_after": [],
        },
        "statement": (
            "The new V2 grounded pipeline reduces hallucination by nullifying unsupported fields "
            "and producing partial profiles when evidence is insufficient."
        ),
    }
    write_json(output_dir / "reports" / "run_report.json", run_report)
    write_json(output_dir / "reports" / "grounded_quality_report.json", quality_report)
    write_json(output_dir / "reports" / "hallucination_reduction_report.json", reduction_report)
    (output_dir / "reports" / "field_confidence_summary.csv").write_text(
        "field_name,field_status,note\n,,no_profiles_processed_due_to_provider_preflight_failed\n",
        encoding="utf-8",
    )
    (output_dir / "reports" / "failed_or_partial_profiles.csv").write_text(
        "profile_file,status,reason\n,failed,provider_preflight_failed\n",
        encoding="utf-8",
    )
    write_json(output_dir / "reports" / "provider_comparison_report.json", {"provider_preflight_status": diagnostic["provider_preflight_status"]})
    return quality_report


def run(args: argparse.Namespace) -> dict[str, Any]:
    configure_logging()
    output_dir = Path(args.output)
    ensure_output_dirs(output_dir)
    os.environ["PROFILE_BUILDER_GROUNDED_PROVIDER_TIMEOUT_SECONDS"] = str(args.provider_timeout)
    os.environ["PROFILE_BUILDER_GROUNDED_OLLAMA_TIMEOUT_SECONDS"] = str(args.ollama_timeout)
    diagnostic = build_provider_diagnostic(args.provider, args.fallback_provider, ollama_timeout_seconds=args.ollama_timeout)
    write_provider_diagnostic_report(output_dir, diagnostic)
    if args.check_providers:
        print_provider_diagnostic(diagnostic)
        return diagnostic
    entries = load_accepted_entries(Path(args.accepted_path))
    limit = args.limit
    if limit is None and not args.run_all:
        limit = 5
    selected = entries[:limit] if limit is not None else entries
    LOGGER.info("Grounded Module 2 V2 starting | accepted=%s | selected=%s", len(entries), len(selected))

    if args.run_all and diagnostic["provider_preflight_status"] == "provider_preflight_failed":
        LOGGER.warning("Provider preflight failed before run-all | action=%s", diagnostic.get("recommended_next_action"))
        return write_preflight_failed_reports(
            output_dir=output_dir,
            accepted_count=len(entries),
            diagnostic=diagnostic,
            compare_with=Path(args.compare_with) if args.compare_with else None,
        )

    existing_artifact_map = build_existing_artifact_map(output_dir)
    processed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    skipped_existing = 0
    checkpoint = {
        "total_accepted": len(entries),
        "already_completed": len(existing_artifact_map),
        "remaining": 0,
        "last_processed_artifact": None,
        "processed_artifacts": [],
        "failed_artifacts": [],
        "skipped_existing": 0,
        "provider_rate_limited": 0,
        "provider_timeout": 0,
        "provider_error": 0,
    }
    to_process: list[dict[str, Any]] = []
    for entry in selected:
        artifact_key = entry.get("artifact_path")
        already_exists = bool(artifact_key and artifact_key in existing_artifact_map)
        if args.resume and already_exists and not args.force:
            skipped_existing += 1
            checkpoint["skipped_existing"] = skipped_existing
            continue
        to_process.append(entry)
    checkpoint["remaining"] = len(to_process)
    write_resume_checkpoint(output_dir, checkpoint)

    for index, entry in enumerate(selected, start=1):
        artifact_key = entry.get("artifact_path")
        if args.resume and artifact_key in existing_artifact_map and not args.force:
            LOGGER.info("Skipping existing grounded profile %s/%s | %s", index, len(selected), artifact_key)
            continue
        LOGGER.info("Processing grounded profile %s/%s | %s", index, len(selected), artifact_key)
        item, failure = process_entry(
            entry,
            output_dir=output_dir,
            provider=args.provider,
            fallback_provider=args.fallback_provider,
            provider_timeout=args.provider_timeout,
            ollama_timeout=args.ollama_timeout,
            max_retries=args.max_retries,
            rate_limit_sleep=args.rate_limit_sleep,
            backoff_multiplier=args.backoff_multiplier,
        )
        if item is not None:
            processed.append(item)
            checkpoint["processed_artifacts"].append(item.get("artifact_path"))
        if failure is not None:
            failed.append(failure)
            checkpoint["failed_artifacts"].append(
                {
                    "artifact_path": failure.get("artifact_path"),
                    "provider_status": failure.get("provider_status"),
                    "error": failure.get("error"),
                }
            )
            provider_status = failure.get("provider_status")
            if provider_status == "provider_rate_limited":
                checkpoint["provider_rate_limited"] += 1
            elif provider_status in {"ollama_timeout", "provider_timeout"}:
                checkpoint["provider_timeout"] += 1
            else:
                checkpoint["provider_error"] += 1
        checkpoint["last_processed_artifact"] = artifact_key
        checkpoint["already_completed"] = len(build_existing_artifact_map(output_dir))
        checkpoint["remaining"] = max(len(to_process) - len(checkpoint["processed_artifacts"]) - len(checkpoint["failed_artifacts"]), 0)
        write_resume_checkpoint(output_dir, checkpoint)
        if args.sleep_between_requests > 0 and index < len(selected):
            time.sleep(args.sleep_between_requests)

    all_processed = collect_output_items(output_dir)
    reports = generate_reports(
        output_dir=output_dir,
        accepted_count=len(entries),
        processed_items=all_processed,
        failed_items=failed,
        provider=args.provider,
        fallback_provider=args.fallback_provider,
        compare_with=Path(args.compare_with) if args.compare_with else None,
        skipped_existing=skipped_existing,
    )
    return reports["grounded_quality_report"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Module 2 V2 grounded structuring on Module 1 accepted CVs.")
    parser.add_argument("--accepted-path", default="data/processed_official_module1/handoff/accepted.json")
    parser.add_argument("--input", default="data/processed_official_module1")
    parser.add_argument("--output", default="data/profile_builder_module2_v2_grounded_all")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Kept for CLI clarity; V2 always writes only isolated preview files.")
    parser.add_argument("--check-providers", action="store_true")
    parser.add_argument("--provider", default="groq_secondary")
    parser.add_argument("--fallback-provider", default="ollama_local")
    parser.add_argument("--provider-timeout", type=float, default=180.0)
    parser.add_argument("--ollama-timeout", type=float, default=120.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--rate-limit-sleep", type=float, default=30.0)
    parser.add_argument("--backoff-multiplier", type=float, default=2.0)
    parser.add_argument("--sleep-between-requests", type=float, default=0.0)
    parser.add_argument("--compare-with", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
