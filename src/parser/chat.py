"""
VM-AI - Chat Testing Interface
Tests add and modify modes with EXP/PRD tag format.
Run: python src/parser/chat.py

Written by: Vanea
"""

import json
import logging
import os
import re
from datetime import datetime
from typing import Dict

import torch
import yaml
from cfg import Config
from regressor import RegressorPredictor
from rule_based_add import parse_add as rule_based_parse
from rule_based_modify import parse_modify_rule_based
from schemas import (
    ALWAYS_EXPLICIT,
    detect_explicit_fields,
    normalize_time,
    pipe_to_schema,
    schema_to_pipe,
)
from transformers import AutoTokenizer, T5ForConditionalGeneration

_uvicorn_log = logging.getLogger("uvicorn")

LOG_FILE = "performance_log.yaml"


def log_entry(mode: str, sentence: str, raw_output: str, parsed_result: Dict):
    """Append one test entry to the performance log."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "input": sentence,
        "raw_output": raw_output,
        "parsed": {
            k: (v["value"] if isinstance(v, dict) else v)
            for k, v in parsed_result.items()
        }
        if "error" not in parsed_result
        else None,
        "error": parsed_result.get("error"),
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        yaml.dump([entry], f, allow_unicode=True, sort_keys=False)
        f.write("\n")


class TaskPlannerPredictor:
    def __init__(self):
        cfg = Config()
        _uvicorn_log.info("Loading T5 model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.output_dir)
        self.model = T5ForConditionalGeneration.from_pretrained(cfg.output_dir)
        self.model.to(self.device)
        self.model.eval()
        self._last_raw_output = ""
        _uvicorn_log.info(f"T5 model ready ({self.device})")

        _uvicorn_log.info("Loading regressor...")
        self.regressor = RegressorPredictor()
        _uvicorn_log.info("Regressor ready")

    _TIME_RE = re.compile(
        r"\b(\d{1,2}:\d{2}|\d{1,2}\s*[ap]m|@\s*\d{1,2})\b", re.IGNORECASE
    )

    def _normalize(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"(\d)(am|pm)", r"\1 \2", text)
        return text

    def _sanity_check(self, schema: Dict, original_sentence: str) -> Dict:
        """Post-generation guard for fixed_time hallucinations."""
        ft = schema.get("fixed_time", {})
        ft_val = ft.get("value") if isinstance(ft, dict) else ft
        if ft_val is True:
            if not self._TIME_RE.search(original_sentence):
                schema["fixed_time"]["value"] = False
                schema["fixed_start"]["value"] = None

        fs = schema.get("fixed_start", {})
        fs_val = fs.get("value") if isinstance(fs, dict) else fs
        if fs_val is not None:
            normalized = normalize_time(str(fs_val))
            if normalized:
                schema["fixed_start"]["value"] = normalized
            else:
                schema["fixed_start"]["value"] = None

        return schema

    def _run_model(self, input_text: str, start_token: str = "name=") -> str:
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to(self.device)

        if start_token:
            decoder_input = self.tokenizer(
                start_token, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(self.device)
        else:
            decoder_input = None

        gen_kwargs = {
            "max_length": 256,
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.1,
            "use_cache": True,
        }
        if decoder_input is not None:
            gen_kwargs["decoder_input_ids"] = decoder_input
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **gen_kwargs,
            )

        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id
        out = [t for t in output_ids[0] if t != pad_id and t != eos_id]
        raw = self.tokenizer.decode(out, skip_special_tokens=False)
        raw = raw.strip()
        self._last_raw_output = raw
        return raw

    def predict_add(self, sentence: str) -> Dict:
        """Add mode: T5 for structure + ML regressor for diff/imp."""
        output = self._run_model(f"add: {sentence}", start_token="name=")
        result = pipe_to_schema(output, input_text=sentence)
        if "error" in result:
            result = rule_based_parse(sentence)
        diff, imp = self.regressor.predict(sentence)
        result["difficulty"] = {"value": f"{diff:.3f}", "predicted": True}
        result["importance"] = {"value": f"{imp:.3f}", "predicted": True}
        log_entry("add", sentence, self._last_raw_output, result)
        return result

    def predict_modify(self, existing_task: Dict, change_prompt: str) -> Dict:
        """Apply changes to existing task.

        Uses ML regressor for importance/difficulty.
        Uses T5 model for other fields (deadline, time, location, etc).
        """
        diff, imp = self.regressor.predict(change_prompt)
        changed = {
            "difficulty": {"value": f"{diff:.3f}", "predicted": True},
            "importance": {"value": f"{imp:.3f}", "predicted": True},
        }

        rule_based = parse_modify_rule_based(change_prompt, existing_task)
        output = self._run_model(change_prompt.lower(), start_token="")
        new_fields = pipe_to_schema(output, input_text=change_prompt)

        if "error" not in new_fields:
            for field, entry in new_fields.items():
                if field in ["importance", "difficulty"]:
                    continue
                if not isinstance(entry, dict):
                    continue
                val = entry.get("value")
                if val is None:
                    continue
                old_entry = existing_task.get(field, {})
                old_val = (
                    old_entry.get("value") if isinstance(old_entry, dict) else old_entry
                )
                if str(val).lower() != str(old_val).lower():
                    changed[field] = entry

        for field in [
            "fixed_time",
            "fixed_start",
            "deadline",
            "start",
            "duration",
            "category",
            "location",
            "recurrent",
            "recurrence_days",
        ]:
            if field in rule_based and field not in changed:
                changed[field] = rule_based[field]

        if not changed:
            result = {"error": "no_changes", "raw": output}
        else:
            result = changed
        log_entry("modify", change_prompt, self._last_raw_output, result)
        return result

    @staticmethod
    def _diff_schemas(old_task: Dict, new_task: Dict) -> Dict:
        """Compare old and new task schemas, return only changed fields."""
        changed = {}
        for field, new_entry in new_task.items():
            new_val = (
                new_entry.get("value") if isinstance(new_entry, dict) else new_entry
            )
            old_entry = old_task.get(field)
            old_val = (
                old_entry.get("value") if isinstance(old_entry, dict) else old_entry
            )

            if new_val is None:
                continue

            old_str = str(old_val).lower() if old_val is not None else ""
            new_str = str(new_val).lower()

            if old_str != new_str:
                changed[field] = {
                    "value": new_val,
                    "predicted": new_entry.get("predicted", False)
                    if isinstance(new_entry, dict)
                    else False,
                }

        return changed


def format_output(result: Dict) -> str:
    if "error" in result:
        return f"   parse failed\n   raw: {result['raw']}"
    if not result:
        return "   nothing extracted"

    fields = [
        "name",
        "start",
        "deadline",
        "difficulty",
        "duration",
        "category",
        "location",
        "importance",
        "fixed_time",
        "fixed_start",
        "recurrent",
        "recurrence_days",
    ]

    rows = []
    for field in fields:
        entry = result.get(field)
        if isinstance(entry, dict):
            value = entry.get("value")
            predicted = entry.get("predicted", False)
        else:
            value = entry
            predicted = False
        value_str = (
            str(value).lower()
            if isinstance(value, bool)
            else (str(value) if value is not None else "-")
        )
        predicted_str = "PRD" if predicted else "EXP"
        rows.append((field, value_str, predicted_str))

    col_field = max(len(r[0]) for r in rows)
    col_value = max(len(r[1]) for r in rows)
    col_pred = max(len(r[2]) for r in rows)
    inner = col_field + col_value + col_pred + 6
    lines = ["+" + "-" * inner + "+"]
    for field, value_str, predicted_str in rows:
        lines.append(
            f"|  {field:<{col_field}}  {value_str:<{col_value}}  {predicted_str:<{col_pred}}  |"
        )
    lines.append("+" + "-" * inner + "+")
    lines.append("  EXP = explicit (user stated) | PRD = predicted (model inferred)")
    return "\n" + "\n".join(lines)


def main():
    print("\n" + "=" * 60)
    print("   VM.AI TASK PLANNER CHAT")
    print("=" * 60)
    print("  add: <prompt>          — extract a new task")
    print("  modify                 — modify last add result")
    print("  modify json            — paste JSON then type change")
    print("  end                    — exit")
    print("=" * 60)
    print(f"  Logging to: {os.path.abspath(LOG_FILE)}")
    print("=" * 60)

    predictor = TaskPlannerPredictor()
    count = 0
    last_result = None

    while True:
        user_input = input(f"\n{count + 1:2d} > ").strip()
        if not user_input:
            continue

        if user_input.lower() == "end":
            print(
                f"\nProcessed {count} inputs — log saved to {os.path.abspath(LOG_FILE)}"
            )
            break

        try:
            if user_input.lower().startswith("add:"):
                sentence = user_input[4:].strip()
                last_result = predictor.predict_add(sentence)
                print(format_output(last_result))
                count += 1

            elif user_input.lower() == "modify json":
                raw = input("   Paste task JSON > ").strip()
                if not raw:
                    continue
                try:
                    pasted = json.loads(raw)
                except json.JSONDecodeError:
                    print("   Invalid JSON")
                    continue
                change = input("   What to change? > ").strip()
                if not change:
                    continue
                changes = predictor.predict_modify(pasted, change)
                print("\n   Changed fields:")
                print(format_output(changes))
                last_result = pasted
                for field, entry in changes.items():
                    if isinstance(entry, dict):
                        last_result[field] = entry
                count += 1

            elif user_input.lower() == "modify":
                if last_result is None or "error" in last_result:
                    print("   No valid task to modify. Run add: first.")
                    continue
                change = input("   What to change? > ").strip()
                if not change:
                    continue
                changes = predictor.predict_modify(last_result, change)
                print("\n   Changed fields:")
                print(format_output(changes))
                for field, entry in changes.items():
                    if field in last_result and isinstance(last_result[field], dict):
                        last_result[field] = entry
                count += 1

            else:
                print("   Start with 'add:' or 'modify:'. Type 'end' to exit.")

        except Exception as e:
            print(f"   Error: {e}")


if __name__ == "__main__":
    main()
