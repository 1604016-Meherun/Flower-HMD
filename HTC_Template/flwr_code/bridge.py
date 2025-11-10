# bridge.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from collections import defaultdict, deque

app = FastAPI()
QUEUES: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20000))

class Sample(BaseModel):
    x: List[float]
    y: int
    ts: float | None = None

class Batch(BaseModel):
    client_id: str
    scene: str
    samples: List[Sample]

@app.post("/batch")
def ingest(batch: Batch):
    q = QUEUES[batch.client_id]
    for s in batch.samples:
        q.append((s.x, s.y))
    return {"ok": True, "queued": len(batch.samples), "client": batch.client_id}
