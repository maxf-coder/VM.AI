"""
VM-AI - YAML Training Data Parser
Parses YAML training data into structured format for training.
Run: imported by other modules

Written by: Vanea
"""

from dataclasses import dataclass, field
from typing import Dict, List

import vars
import yaml


@dataclass
class VMAI_YamlTrainingParsedData:
    label_list: List[str]
    templates: List[str]
    tasks: List[str]
    durations: List[str]
    deadlines: List[str]
    locations: List[str] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)
    times: List[str] = field(default_factory=list)
    priorities: List[str] = field(default_factory=list)
    difficulties: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    fixed_starts: List[str] = field(default_factory=list)
    recurrence_days: List[str] = field(default_factory=list)

    label2id: Dict[str, int] = field(init=False)
    id2label: Dict[int, str] = field(init=False)

    def get_placeholder_map(self) -> Dict[str, List[str]]:
        mapping = {
            "TASK": self.tasks,
            "DURATION": self.durations,
            "DEADLINE": self.deadlines,
            "LOCATION": self.locations,
            "DATE": self.dates,
            "TIME": self.times,
            "PRIORITY": self.priorities,
            "DIFFICULTY": self.difficulties,
            "CATEGORY": self.categories,
            "FIXED_START": self.fixed_starts,
            "RECURRENCE_DAY": self.recurrence_days,
        }
        return {k: v for k, v in mapping.items() if v}

    def __post_init__(self):
        self.label2id = {label: i for i, label in enumerate(self.label_list)}
        self.id2label = {i: label for label, i in self.label2id.items()}

    def print_nice(self):
        print("\n" + "=" * 60)
        print("VMAI TRAINING CONFIGURATION")
        print("=" * 60)

        fields_to_print = [
            "templates",
            "tasks",
            "durations",
            "deadlines",
            "locations",
            "dates",
            "times",
            "priorities",
            "difficulties",
            "categories",
            "fixed_starts",
            "recurrence_days",
        ]

        for field_name in fields_to_print:
            if hasattr(self, field_name):
                values = getattr(self, field_name)
                if values:
                    icon = "[UNKNOWN]" if field_name == "templates" else ""
                    print(f"\n{icon} {field_name.upper()} ({len(values)} items):")
                    for i, v in enumerate(values, 1):
                        print(f"  {i}. {v}")
        print("\n" + "=" * 60)


class VMAI_YamlParser:
    def __init__(self, yaml_file: str):
        self.yaml_file = yaml_file
        self.data = None

    def load_yaml(self):
        with open(self.yaml_file, "r", encoding="utf-8") as file:
            self.data = yaml.safe_load(file)

    def parse(self) -> VMAI_YamlTrainingParsedData:
        if not self.data:
            raise ValueError("YAML data not loaded")

        return VMAI_YamlTrainingParsedData(
            label_list=self.data.get("labels", []),
            templates=self.data.get("templates", []),
            tasks=self.data.get("tasks", []),
            durations=self.data.get("durations", []),
            deadlines=self.data.get("deadlines", []),
            locations=self.data.get("locations", []),
            dates=self.data.get("dates", []),
            times=self.data.get("times", []),
            priorities=self.data.get("priorities", []),
            difficulties=self.data.get("difficulties", []),
            categories=self.data.get("categories", []),
            fixed_starts=self.data.get("fixed_starts", []),
            recurrence_days=self.data.get("recurrence_days", []),
        )


class VMAI_RealDataParser:
    def __init__(self, yaml_file: str):
        self.yaml_file = yaml_file
        self.data = None

    def load_yaml(self):
        with open(self.yaml_file, "r", encoding="utf-8") as file:
            self.data = yaml.safe_load(file)

    def parse(self) -> list:
        if not self.data:
            raise ValueError("YAML data not loaded")
        return self.data.get("examples", [])


if __name__ == "__main__":
    parser = VMAI_YamlParser(f"./data/{vars.SYNTHETIC_DATASET}")
    parser.load_yaml()
    parsed_data = parser.parse()
    parsed_data.print_nice()
