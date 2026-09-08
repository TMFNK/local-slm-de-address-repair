"""Local SLM German address repair: rules floor, schema, audit, scorer."""

from addr_repair.audit import write_audit_record
from addr_repair.prompts import PRODUCTION_PROMPT_REV, build_repair_prompt
from addr_repair.rules import repair_with_rules
from addr_repair.schema import OUTPUT_SCHEMA, validate_output
from addr_repair.scorer import score_records

__all__ = [
    "OUTPUT_SCHEMA",
    "PRODUCTION_PROMPT_REV",
    "build_repair_prompt",
    "repair_with_rules",
    "score_records",
    "validate_output",
    "write_audit_record",
]
