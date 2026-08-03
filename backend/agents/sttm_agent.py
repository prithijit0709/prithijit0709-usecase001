import json
import re
import uuid
from typing import Any

from core.audit import AuditLogger
from core.config import GEMINI_MODEL, GOOGLE_API_KEY, GROQ_API_KEY, GROQ_MODEL, LLM_PROVIDER

try:
    from core.memory import query_memory
except Exception:
    def query_memory(query_text: str, n_results: int = 5) -> list[dict[str, Any]]:
        return []


class BaseAgent:
    """Minimal base agent abstraction for LLM access, logging, and context retrieval."""

    def __init__(self, agent_name: str, run_id: str | None = None, llm: Any | None = None):
        self.agent_name = agent_name
        self.run_id = run_id or str(uuid.uuid4())
        self.audit = AuditLogger(self.run_id)
        self.llm = llm or self._make_llm()

    def _make_llm(self) -> Any:
        if LLM_PROVIDER == "groq":
            from langchain_groq import ChatGroq

            return ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL)

        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, model=GEMINI_MODEL)

    def log(self, action: str, **kwargs) -> None:
        self.audit.log(self.agent_name, action, **kwargs)

    def invoke_llm(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else response
        return content if isinstance(content, str) else json.dumps(content, default=str)

    def get_ontology_context(
        self,
        layer: str,
        tables: list[Any],
        profiles: dict[str, Any],
        additional_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if additional_context and isinstance(additional_context.get("ontology"), dict):
            return additional_context["ontology"]
        return {
            "layer": layer,
            "known_entities": sorted({self._table_name(table, i) for i, table in enumerate(tables)}),
            "profile_keys": sorted(list(profiles.keys())),
        }

    def retrieve_knowledge(self, query: str, n_results: int = 4) -> list[dict[str, Any]]:
        return query_memory(query, n_results=n_results)

    @staticmethod
    def _table_name(table: Any, index: int) -> str:
        if isinstance(table, str) and table.strip():
            return table.strip()
        if isinstance(table, dict):
            for key in ("name", "table", "table_name", "source_table", "target"):
                value = table.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return f"table_{index + 1}"


class STTMAgent(BaseAgent):
    """Proposal-only STTM agent preserving human approval gates."""

    VALID_LAYERS = {"bronze", "silver", "gold"}

    def __init__(self, run_id: str | None = None, llm: Any | None = None):
        super().__init__(agent_name="sttm_agent", run_id=run_id, llm=llm)

    def run(
        self,
        layer: str,
        tables: list[Any],
        profiles: dict[str, Any],
        intent: str,
        additional_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._validate_layer(layer)
        if not isinstance(tables, list) or not tables:
            raise ValueError("tables must be a non-empty list")
        if not isinstance(profiles, dict):
            raise ValueError("profiles must be a dictionary")
        if not isinstance(intent, str) or not intent.strip():
            raise ValueError("intent must be a non-empty string")
        if additional_context is not None and not isinstance(additional_context, dict):
            raise ValueError("additional_context must be a dictionary when provided")

        self.log("run_started", layer=layer, table_count=len(tables), intent=intent[:200])
        ontology_context = self.get_ontology_context(layer, tables, profiles, additional_context)
        context_docs = self._collect_context_docs(layer, intent, tables, additional_context)

        rules: list[dict[str, Any]] = []
        fallback_count = 0

        for index, table in enumerate(tables):
            name = self._table_name(table, index)
            self.log("proposal_started", layer=layer, name=name)
            try:
                if layer == "bronze":
                    rule = self._generate_bronze_rules(
                        table=table,
                        table_name=name,
                        profiles=profiles,
                        intent=intent,
                        ontology_context=ontology_context,
                        context_docs=context_docs,
                        additional_context=additional_context,
                    )
                elif layer == "silver":
                    rule = self._generate_silver_rules(
                        table=table,
                        table_name=name,
                        profiles=profiles,
                        intent=intent,
                        ontology_context=ontology_context,
                        context_docs=context_docs,
                        additional_context=additional_context,
                    )
                else:
                    rule = self._generate_gold_rules(
                        table=table,
                        target_name=name,
                        profiles=profiles,
                        intent=intent,
                        ontology_context=ontology_context,
                        context_docs=context_docs,
                        additional_context=additional_context,
                    )
            except Exception as exc:
                fallback_count += 1
                self.log("proposal_failed", layer=layer, name=name, error=str(exc))
                if layer == "gold":
                    rule = self._fallback_for_gold(name, reason=str(exc))
                else:
                    rule = self._fallback_for_table(layer, name, table, reason=str(exc))
            rules.append(rule)
            self.log("proposal_completed", layer=layer, name=name, fallback=bool(rule.get("fallback", False)))

        summary = {
            "message": "Proposal-only STTM output. No transformations executed. Approval required.",
            "generated": len(rules),
            "fallbacks": fallback_count,
            "approval_required": True,
            "execution_performed": False,
            "approved": False,
        }
        result = {
            "layer": layer,
            "status": "Proposed",
            "rules": rules,
            "summary": summary,
        }
        self.log("run_completed", layer=layer, generated=len(rules), fallbacks=fallback_count)
        return result

    def _validate_layer(self, layer: str) -> None:
        if layer not in self.VALID_LAYERS:
            raise ValueError(f"Invalid layer '{layer}'. Valid layers: bronze, silver, gold")

    def _collect_context_docs(
        self,
        layer: str,
        intent: str,
        tables: list[Any],
        additional_context: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        docs = self.retrieve_knowledge(query=f"{layer} {intent}")
        if additional_context and isinstance(additional_context.get("knowledge"), list):
            for idx, item in enumerate(additional_context["knowledge"]):
                docs.append({"id": f"extra_{idx}", "document": str(item), "metadata": {"source": "additional_context"}})
        table_names = [self._table_name(t, i) for i, t in enumerate(tables)]
        self.log("context_collected", layer=layer, knowledge_docs=len(docs), tables=table_names)
        return docs[:8]

    def _generate_bronze_rules(
        self,
        table: Any,
        table_name: str,
        profiles: dict[str, Any],
        intent: str,
        ontology_context: dict[str, Any],
        context_docs: list[dict[str, Any]],
        additional_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        prompt = self._build_bronze_prompt(
            table_name=table_name,
            table=table,
            profiles=profiles,
            intent=intent,
            ontology_context=ontology_context,
            context_docs=context_docs,
            additional_context=additional_context,
        )
        raw = self.invoke_llm(prompt)
        return self._parse_or_normalize_response(raw=raw, layer="bronze", entity_name=table_name)

    def _generate_silver_rules(
        self,
        table: Any,
        table_name: str,
        profiles: dict[str, Any],
        intent: str,
        ontology_context: dict[str, Any],
        context_docs: list[dict[str, Any]],
        additional_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        prompt = self._build_silver_prompt(
            table_name=table_name,
            table=table,
            profiles=profiles,
            intent=intent,
            ontology_context=ontology_context,
            context_docs=context_docs,
            additional_context=additional_context,
        )
        raw = self.invoke_llm(prompt)
        return self._parse_or_normalize_response(raw=raw, layer="silver", entity_name=table_name)

    def _generate_gold_rules(
        self,
        table: Any,
        target_name: str,
        profiles: dict[str, Any],
        intent: str,
        ontology_context: dict[str, Any],
        context_docs: list[dict[str, Any]],
        additional_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        prompt = self._build_gold_prompt(
            target_name=target_name,
            table=table,
            profiles=profiles,
            intent=intent,
            ontology_context=ontology_context,
            context_docs=context_docs,
            additional_context=additional_context,
        )
        raw = self.invoke_llm(prompt)
        return self._parse_or_normalize_response(raw=raw, layer="gold", entity_name=target_name)

    def _build_bronze_prompt(
        self,
        table_name: str,
        table: Any,
        profiles: dict[str, Any],
        intent: str,
        ontology_context: dict[str, Any],
        context_docs: list[dict[str, Any]],
        additional_context: dict[str, Any] | None,
    ) -> str:
        schema = {
            "table": table_name,
            "status": "Proposed",
            "rationale": "string",
            "proposals": [
                {
                    "source_column": "string",
                    "target_column": "string",
                    "action": "pass_through|cast|standardize",
                    "proposed_type": "string",
                    "standardization": "string",
                    "reason": "string",
                    "confidence_score": 0.0,
                    "risk_level": "Low|Medium|High",
                }
            ],
        }
        return (
            "You generate SOURCE-TO-TARGET MAPPING PROPOSALS only for bronze. "
            "Do not execute transformations. Do not approve anything. "
            "Output JSON only, no markdown. "
            "Focus on source fidelity, conservative casting, light standardization. "
            f"Intent: {intent}\n"
            f"Table name: {table_name}\n"
            f"Table metadata: {json.dumps(table, default=str)}\n"
            f"Profiles: {json.dumps(profiles, default=str)[:5000]}\n"
            f"Ontology: {json.dumps(ontology_context, default=str)[:2000]}\n"
            f"Knowledge: {json.dumps(context_docs, default=str)[:2000]}\n"
            f"Additional context: {json.dumps(additional_context or {}, default=str)[:2000]}\n"
            f"Return exactly one JSON object with schema: {json.dumps(schema)}"
        )

    def _build_silver_prompt(
        self,
        table_name: str,
        table: Any,
        profiles: dict[str, Any],
        intent: str,
        ontology_context: dict[str, Any],
        context_docs: list[dict[str, Any]],
        additional_context: dict[str, Any] | None,
    ) -> str:
        schema = {
            "table": table_name,
            "status": "Proposed",
            "rationale": "string",
            "proposals": [
                {
                    "column": "string",
                    "deduplication": "string",
                    "null_handling": "string",
                    "normalization": "string",
                    "outlier_handling": "string",
                    "validation_rule": "string",
                    "reason": "string",
                    "confidence_score": 0.0,
                    "risk_level": "Low|Medium|High",
                }
            ],
        }
        return (
            "You generate SOURCE-TO-TARGET MAPPING PROPOSALS only for silver. "
            "Do not execute transformations. Do not approve anything. "
            "Output JSON only, no markdown. "
            "Include deduplication, null handling, normalization, outlier handling, validation-style rules. "
            f"Intent: {intent}\n"
            f"Table name: {table_name}\n"
            f"Table metadata: {json.dumps(table, default=str)}\n"
            f"Profiles: {json.dumps(profiles, default=str)[:5000]}\n"
            f"Ontology: {json.dumps(ontology_context, default=str)[:2000]}\n"
            f"Knowledge: {json.dumps(context_docs, default=str)[:2000]}\n"
            f"Additional context: {json.dumps(additional_context or {}, default=str)[:2000]}\n"
            f"Return exactly one JSON object with schema: {json.dumps(schema)}"
        )

    def _build_gold_prompt(
        self,
        target_name: str,
        table: Any,
        profiles: dict[str, Any],
        intent: str,
        ontology_context: dict[str, Any],
        context_docs: list[dict[str, Any]],
        additional_context: dict[str, Any] | None,
    ) -> str:
        schema = {
            "target": target_name,
            "status": "Proposed",
            "sources": ["string"],
            "join_plan": [{"left": "string", "right": "string", "condition": "string", "join_type": "string"}],
            "dimensions": [{"name": "string", "expression": "string"}],
            "metrics": [{"name": "string", "calculation": "string", "kpi": "string"}],
            "filters": ["string"],
            "rationale": "string",
            "confidence_score": 0.0,
            "risk_level": "Low|Medium|High",
        }
        return (
            "You generate SOURCE-TO-TARGET MAPPING PROPOSALS only for gold. "
            "Do not execute transformations. Do not approve anything. "
            "Do not output executable code. "
            "Output JSON only, no markdown. "
            "Align mappings to business intent with sources, joins, dimensions, metrics, KPIs, and rationale. "
            f"Business intent: {intent}\n"
            f"Target name: {target_name}\n"
            f"Target metadata: {json.dumps(table, default=str)}\n"
            f"Profiles: {json.dumps(profiles, default=str)[:5000]}\n"
            f"Ontology: {json.dumps(ontology_context, default=str)[:2000]}\n"
            f"Knowledge: {json.dumps(context_docs, default=str)[:2000]}\n"
            f"Additional context: {json.dumps(additional_context or {}, default=str)[:2000]}\n"
            f"Return exactly one JSON object with schema: {json.dumps(schema)}"
        )

    def _parse_or_normalize_response(self, raw: str, layer: str, entity_name: str) -> dict[str, Any]:
        parsed = self._extract_json_object(raw)
        if not isinstance(parsed, dict):
            raise ValueError("LLM response did not contain a valid JSON object")

        if layer == "gold":
            return self._normalize_gold(parsed, entity_name)
        return self._normalize_table_rule(layer, parsed, entity_name)

    def _normalize_table_rule(self, layer: str, payload: dict[str, Any], table_name: str) -> dict[str, Any]:
        proposals = payload.get("proposals")
        if not isinstance(proposals, list):
            proposals = []

        normalized_proposals: list[dict[str, Any]] = []
        for proposal in proposals:
            if not isinstance(proposal, dict):
                continue
            confidence = self._coerce_confidence(proposal.get("confidence_score"))
            risk = str(proposal.get("risk_level", "Medium")).title()
            if risk not in {"Low", "Medium", "High"}:
                risk = "Medium"
            proposal_copy = dict(proposal)
            proposal_copy["confidence_score"] = confidence
            proposal_copy["risk_level"] = risk
            proposal_copy["status"] = "Proposed"
            normalized_proposals.append(proposal_copy)

        if not normalized_proposals:
            return self._fallback_for_table(layer, table_name, {}, reason="No valid proposals from model")

        return {
            "table": str(payload.get("table") or table_name),
            "status": "Proposed",
            "rationale": str(payload.get("rationale") or "Proposal-only mapping for human review"),
            "proposals": normalized_proposals,
        }

    def _normalize_gold(self, payload: dict[str, Any], target_name: str) -> dict[str, Any]:
        sources = payload.get("sources") if isinstance(payload.get("sources"), list) else []
        join_plan = payload.get("join_plan") if isinstance(payload.get("join_plan"), list) else []
        dimensions = payload.get("dimensions") if isinstance(payload.get("dimensions"), list) else []
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), list) else []
        filters = payload.get("filters") if isinstance(payload.get("filters"), list) else []

        if not sources and isinstance(payload.get("source"), str):
            sources = [payload["source"]]

        confidence = self._coerce_confidence(payload.get("confidence_score"))
        risk = str(payload.get("risk_level", "Medium")).title()
        if risk not in {"Low", "Medium", "High"}:
            risk = "Medium"

        normalized = {
            "target": str(payload.get("target") or target_name),
            "status": "Proposed",
            "sources": [str(item) for item in sources],
            "join_plan": [item for item in join_plan if isinstance(item, dict)],
            "dimensions": [item for item in dimensions if isinstance(item, dict)],
            "metrics": [item for item in metrics if isinstance(item, dict)],
            "filters": [str(item) for item in filters],
            "rationale": str(payload.get("rationale") or "Analytical proposal for human review"),
            "confidence_score": confidence,
            "risk_level": risk,
        }
        if not normalized["sources"] and not normalized["metrics"] and not normalized["dimensions"]:
            return self._fallback_for_gold(target_name, "Insufficient structured gold proposal")
        return normalized

    def _extract_json_object(self, raw: str) -> Any:
        if not isinstance(raw, str):
            return raw

        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()

        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass

        for start_char, end_char in (("{", "}"), ("[", "]")):
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start == -1 or end == -1 or end <= start:
                continue
            candidate = text[start : end + 1]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                    return parsed[0]
                return parsed
            except (json.JSONDecodeError, TypeError):
                continue
        return None

    def _fallback_for_table(self, layer: str, table_name: str, table: Any, reason: str) -> dict[str, Any]:
        if layer == "silver":
            proposal = {
                "column": "*",
                "deduplication": "none",
                "null_handling": "preserve",
                "normalization": "minimal_trim",
                "outlier_handling": "none",
                "validation_rule": "non_destructive_review_required",
                "reason": f"Fallback used: {reason}",
                "confidence_score": 0.35,
                "risk_level": "Medium",
                "status": "Proposed",
            }
        else:
            proposal = {
                "source_column": "*",
                "target_column": "*",
                "action": "pass_through",
                "proposed_type": "preserve",
                "standardization": "minimal",
                "reason": f"Fallback used: {reason}",
                "confidence_score": 0.4,
                "risk_level": "Medium",
                "status": "Proposed",
            }

        self.log("fallback_used", layer=layer, table=table_name, reason=reason)
        return {
            "table": table_name,
            "status": "Proposed",
            "fallback": True,
            "fallback_reason": reason,
            "rationale": "Safe passthrough proposal requiring human review",
            "proposals": [proposal],
            "source_snapshot": table if isinstance(table, dict) else {"name": table_name},
        }

    def _fallback_for_gold(self, target_name: str, reason: str) -> dict[str, Any]:
        self.log("fallback_used", layer="gold", target=target_name, reason=reason)
        return {
            "target": target_name,
            "status": "Proposed",
            "fallback": True,
            "fallback_reason": reason,
            "sources": [],
            "join_plan": [],
            "dimensions": [],
            "metrics": [],
            "filters": [],
            "rationale": "No-op structured proposal due to generation failure. Human review required.",
            "confidence_score": 0.2,
            "risk_level": "High",
        }

    def _coerce_confidence(self, value: Any) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.5
        return max(0.0, min(1.0, confidence))
