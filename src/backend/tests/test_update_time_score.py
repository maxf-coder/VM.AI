"""
Manual test: update_time_score
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import SessionLocal
from app.services.stats_recorder import stats_recorder

db = SessionLocal()

task_id = input("task_id: ")
slot_start = input("slot_start (YYYY-MM-DD HH:MM): ")
boost = float(input("boost: "))

from datetime import datetime
slot_dt = datetime.strptime(slot_start, "%Y-%m-%d %H:%M")

result = stats_recorder.update_time_score(db, task_id, slot_dt, boost)
print("Result:", result)
