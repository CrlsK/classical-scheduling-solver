import json
with open('/sessions/sweet-bold-newton/.push_files_data.json') as f:
    d = json.load(f)
print(d[1]['content'])