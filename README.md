Foxglove Visualization Demo (Python + uv)

This project publishes a rich set of Foxglove visualization streams using the official
`foxglove` Python SDK with proper schema types and coordinate frames.

Features
- Uses **Pydantic** models to define JSON schemas for custom message types
- Uses **foxglove.schemas** for well-known types (SceneUpdate, FrameTransforms, PointCloud, etc.)
- Uses **typed channels** (FrameTransformsChannel, SceneUpdateChannel, etc.) for proper encoding
- Full **coordinate frame tree**: `world` → `robot` → `camera_link`
- Supports live visualization via Foxglove Studio
- Records data to MCAP files for later playback

Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/plot/sample` | JSON (Pydantic) | Time-series signals: sine, cosine, saw, noise |
| `/custom/status` | JSON (Pydantic) | Device status with battery, temperatures, events |
| `/tf` | FrameTransforms | Coordinate transforms: world→robot (animated), robot→camera_link (fixed) |
| `/viz/scene` | SceneUpdate | 3D primitives: spinning cube, sphere, line strip |
| `/viz/point_cloud` | PointCloud | Animated ring of colored points |
| `/viz/image` | RawImage | 160x120 RGB gradient pattern |
| `/viz/camera_info` | CameraCalibration | Pinhole camera model matching the image |
| `/system/log` | Log | Info/warning log messages |

Coordinate Frames

The transform tree enables proper 3D visualization:

    world
      └── robot (animated: circles around origin)
            └── camera_link (fixed: 30cm in front of robot)

Quick start

1) Install deps with uv:
   uv sync

2) Run the publisher:
   uv run python main.py --rate 10

3) Open Foxglove Studio and connect:
   ws://localhost:8765

4) Optional: override MCAP output path:
   uv run python main.py --mcap demo.mcap

5) Optional: disable MCAP recording:
   uv run python main.py --no-mcap

Suggested panels in Foxglove Studio

- **Plot**: chart `/plot/sample.sine`, `/plot/sample.cosine`, `/plot/sample.saw`
- **3D**: visualize the scene with proper transforms
  - Add `/viz/scene` to see animated cube, sphere, and lines
  - Add `/viz/point_cloud` to see the ring of points
  - Enable "Follow" mode to track the robot frame
- **Image**: view `/viz/image` with `/viz/camera_info` for calibration
- **Transform Tree**: inspect the `/tf` topic to see world→robot→camera_link
- **Log**: inspect `/system/log`
- **Raw Messages**: inspect `/custom/status` (custom JSON schema)

Custom Schemas

Two Pydantic models define custom JSON schemas:

- `PlotSample` - Time-series data with t, sine, cosine, saw, and noise fields
- `CustomStatus` - Device status with device_id, state, battery, temperatures, and events

Pydantic automatically generates JSON Schema which Foxglove uses for
schema validation and message visualization.

Implementation Notes

- Uses **protobuf-encoded** messages for well-known Foxglove types (more efficient than JSON)
- **FrameTransforms** published on `/tf` enables proper 3D coordinate frame tracking
- **CameraCalibration** includes intrinsic matrix (K), projection matrix (P), and distortion params
- Channels must be created before MCAP writer to be recorded
- By default, records to `foxglove-demo.mcap` with `allow_overwrite=True`
