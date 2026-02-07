# Pieskieo Python SDK

Simple sync client for Pieskieo HTTP API.

## Install (editable)
```
pip install -e .
```

## Usage
```python
from pieskieo import PieskieoClient

c = PieskieoClient("http://localhost:8000")

vec_id = c.put_vector([0.1,0.2,0.3], meta={"type":"demo"})
hits = c.search([0.1,0.2,0.3], k=3)
print(hits)
```
