Foxglove Visualization Demo (Python + uv)

This project publishes a rich set of Foxglove WebSocket streams: time-series plots,
3D scene objects, point clouds, images, transforms, logs, and custom schemas.

Quick start
1) Install deps with uv:
   uv sync
2) Run the publisher:
   uv run python main.py --rate 10
3) Open Foxglove Studio and connect:
   ws://localhost:8765

Suggested panels in Foxglove Studio
- Plot: chart `plot/sample.sine`, `plot/sample.cosine`, `plot/sample.saw`
- 3D: visualize `viz/scene`, `viz/point_cloud`, `viz/pose`, `viz/transform`
- Image: view `viz/image`
- Log: inspect `system/log`
- Raw Messages: inspect `custom/status` (custom JSON schema)

Notes
- The demo uses JSON encoding with the Foxglove WebSocket server.
- `plot/sample` and `custom/status` include explicit JSON schemas.
