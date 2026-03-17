import json
import struct 

struct_format = "<QQI4x"

tma = open("tma_profile.bin", "rb").read()
mm = open("mm_profile.bin", "rb").read()

tma_events = []
for i in range(0, len(tma), struct.calcsize(struct_format)):
    event = struct.unpack(struct_format, tma[i:i+struct.calcsize(struct_format)])
    if event == (0, 0, 0):break
    tma_events.append(event)

mm_events = []
for i in range(0, len(mm), struct.calcsize(struct_format)):
    event = struct.unpack(struct_format, mm[i:i+struct.calcsize(struct_format)])
    if event == (0, 0, 0):break
    mm_events.append(event)

perfetto_trace = []

for i in range(len(tma_events)):
    if i < 2: continue
    perfetto_trace.append({
        "name": f"tma{i}",
        "cat": "PERF",
        "ph": "b",
        "ts": tma_events[i][0],
        "id": f"tma{i}",
        "pid": 0,
        "tid": 3,
    })
    perfetto_trace.append({
        "name": f"tma{i}",
        "cat": "PERF",
        "ph": "e",
        "id": f"tma{i}",
        "ts": tma_events[i][1],
        "pid": 0,
        "tid": 3,
    })
    print(f"tma{i}", "begin : ", tma_events[i][0], "end : ", tma_events[i][1], "duration : ", tma_events[i][1] - tma_events[i][0])

for i in range(len(mm_events)):
    perfetto_trace.append({
        "name": f"mm{i}",
        "cat": "PERF",
        "ph": "b",
        "ts": mm_events[i][0],
        "id": f"mm{i}",
        "pid": 0,
        "tid": 4,
    })
    perfetto_trace.append({
        "name": f"mm{i}",
        "cat": "PERF",
        "ph": "e",
        "ts": mm_events[i][1],
        "id": f"mm{i}",
        "pid": 0,
        "tid": 4,
    })
    print(f"mm{i}", "begin : ", mm_events[i][0], "end : ", mm_events[i][1], "duration : ", mm_events[i][1] - mm_events[i][0])
 
with open("perfetto_trace.json", "w") as f:
    json.dump(perfetto_trace, f)