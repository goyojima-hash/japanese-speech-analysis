"""One execution path for fact-only speaking-axis modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from .accuracy import AccuracyModule
from .coherence import CoherenceModule
from .evidence.models import EvidenceBundle
from .range import RangeExtractor


SUPPORTED_FACT_MODULES = frozenset({"fluency", "range", "accuracy", "coherence"})
DEFAULT_FACT_MODULES = SUPPORTED_FACT_MODULES
INTERACTIVE_FACT_MODULES = SUPPORTED_FACT_MODULES


@dataclass(frozen=True)
class FactModuleRun:
    """Selected fact packets, reusable by every entry point."""

    active_modules: frozenset[str]
    packets: Mapping[str, dict[str, Any]]

    def merge_objective_data(self, base_data: Mapping[str, Any]) -> dict[str, Any]:
        return {**base_data, **self.packets}

    def add_packets_to_roleplay_input(self, base_input: Mapping[str, Any]) -> dict[str, Any]:
        return {**base_input, **self.packets}


def run_fact_modules(
    evidence: EvidenceBundle,
    *,
    selected_modules: Iterable[str] | None = None,
    range_extractor: RangeExtractor | None = None,
    on_module_start: Callable[[str], None] | None = None,
    on_module_result: Callable[[str, dict[str, Any]], None] | None = None,
) -> FactModuleRun:
    """Collect selected packets from one already-built EvidenceBundle."""

    active_modules = normalize_fact_modules(selected_modules)
    packets: dict[str, dict[str, Any]] = {}
    if "range" in active_modules:
        _notify_start(on_module_start, "range")
        active_range_extractor = range_extractor or RangeExtractor.default()
        analyze_evidence = getattr(type(active_range_extractor), "analyze_linguistic_evidence", None)
        packets["range_data"] = (
            analyze_evidence(active_range_extractor, evidence.linguistic)
            if callable(analyze_evidence)
            else active_range_extractor.analyze(evidence.speech.raw_transcript_hiragana)
        )
        _notify_result(on_module_result, "range", packets["range_data"])
    if "accuracy" in active_modules:
        _notify_start(on_module_start, "accuracy")
        packets["accuracy_data"] = AccuracyModule().collect(evidence).to_dict()
        _notify_result(on_module_result, "accuracy", packets["accuracy_data"])
    if "coherence" in active_modules:
        _notify_start(on_module_start, "coherence")
        packets["coherence_data"] = CoherenceModule().collect(evidence).to_dict()
        _notify_result(on_module_result, "coherence", packets["coherence_data"])
    return FactModuleRun(active_modules=active_modules, packets=packets)


def normalize_fact_modules(selected_modules: Iterable[str] | None) -> frozenset[str]:
    modules = DEFAULT_FACT_MODULES if selected_modules is None else frozenset(selected_modules)
    unknown = modules - SUPPORTED_FACT_MODULES
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unsupported fact module(s): {names}")
    return modules


def _notify_start(callback: Callable[[str], None] | None, module_id: str) -> None:
    if callback is not None:
        callback(module_id)


def _notify_result(
    callback: Callable[[str, dict[str, Any]], None] | None,
    module_id: str,
    packet: dict[str, Any],
) -> None:
    if callback is not None:
        callback(module_id, packet)
