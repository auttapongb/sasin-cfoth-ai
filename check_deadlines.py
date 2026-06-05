#!/usr/bin/env python3
"""
Deadline Warning Script - Check all course deadlines and generate warnings
Run daily via cron
"""
import json
from datetime import datetime, date
import sys

COURSES_FILE = "/data/sasin-cfoth-ai/courses.json"
WARNINGS_FILE = "/data/sasin-cfoth-ai/deadline_warnings.txt"

try:
    with open(COURSES_FILE) as f:
        courses = json.load(f)
except Exception as e:
    print(f"ERROR loading courses: {e}")
    sys.exit(1)

today = date.today()
warnings = []

for code, course in courses.items():
    for dl in course.get("deadlines", []):
        dl_date = datetime.strptime(dl["date"], "%Y-%m-%d").date()
        days_left = (dl_date - today).days
        
        if days_left < 0:
            continue
        
        if days_left <= 2:
            urgency = "URGENT"
        elif days_left <= 5:
            urgency = "SOON"
        else:
            urgency = "UPCOMING"
        
        warn = {
            "course": code,
            "days_left": days_left,
            "date": dl["date"],
            "label": dl["label"],
            "urgency": urgency,
            "type": dl.get("type", "unknown")
        }
        warnings.append(warn)

warnings.sort(key=lambda w: w["days_left"])

lines = []
header = "=== DEADLINE WARNINGS - {} ===".format(today)
lines.append(header)
lines.append("")
lines.append("Total upcoming deadlines: {}".format(len(warnings)))
lines.append("")

for w in warnings:
    if w["days_left"] == 0:
        lines.append("[URGENT] **TODAY!** [{}] {}".format(w["course"], w["label"]))
    elif w["days_left"] == 1:
        lines.append("[URGENT] **TOMORROW!** [{}] {}".format(w["course"], w["label"]))
    else:
        lines.append("[{}] {}d left - [{}] {} ({})".format(
            w["urgency"], w["days_left"], w["course"], w["label"], w["date"]))
    lines.append("")

output = "\n".join(lines)
print(output)

with open(WARNINGS_FILE, "w") as f:
    f.write(output)
