import argparse
import base64
import math
import random
import time
from typing import Any, Dict, List

import foxglove
import foxglove.schemas as schemas
from foxglove.channels import (
    CameraCalibrationChannel,
    FrameTransformsChannel,
    LogChannel,
    PointCloudChannel,
    RawImageChannel,
    SceneUpdateChannel,
)
from pydantic import BaseModel


# =============================================================================
# Pydantic Models for Custom Schemas
# =============================================================================


class PlotSample(BaseModel):
    """Custom time-series plot sample."""

    t: float
    sine: float
    cosine: float
    saw: float
    noise: float


class CustomStatus(BaseModel):
    """Custom device status message."""

    device_id: str
    state: str
    battery: float
    temperatures: List[float]
    events: List[dict]  # Use dict to avoid $ref in JSON schema


# =============================================================================
# Utility Functions
# =============================================================================


def now_sec_nsec() -> Dict[str, int]:
    ts = time.time_ns()
    return {"sec": ts // 1_000_000_000, "nsec": ts % 1_000_000_000}


def quat_from_yaw(yaw: float) -> schemas.Quaternion:
    """Convert yaw angle to Quaternion."""
    half = yaw * 0.5
    return schemas.Quaternion(
        x=0.0,
        y=0.0,
        z=math.sin(half),
        w=math.cos(half),
    )


def timestamp_now() -> schemas.Timestamp:
    """Get current timestamp as Foxglove Timestamp."""
    ts = time.time_ns()
    return schemas.Timestamp(sec=ts // 1_000_000_000, nsec=ts % 1_000_000_000)


# =============================================================================
# Message Builders
# =============================================================================


def encode_image_rgb(width: int, height: int, t: float) -> schemas.RawImage:
    """Build a RawImage with gradient pattern."""
    data = bytearray()
    for y in range(height):
        for x in range(width):
            r = int((x / max(1, width - 1)) * 255)
            g = int((y / max(1, height - 1)) * 255)
            b = int(((math.sin(t + x * 0.1) + 1.0) * 0.5) * 255)
            data.extend((r, g, b))
    return schemas.RawImage(
        timestamp=timestamp_now(),
        frame_id="camera",
        height=height,
        width=width,
        encoding="rgb8",
        step=width * 3,
        data=bytes(data),
    )


def build_camera_calibration(width: int, height: int) -> schemas.CameraCalibration:
    """Build CameraCalibration for the camera."""
    # Simple pinhole camera model
    fx = fy = max(width, height)  # focal length in pixels
    cx = width / 2.0  # principal point x
    cy = height / 2.0  # principal point y
    return schemas.CameraCalibration(
        timestamp=timestamp_now(),
        frame_id="camera_link",
        height=height,
        width=width,
        distortion_model="plumb_bob",
        D=[0.0, 0.0, 0.0, 0.0, 0.0],  # No distortion
        K=[fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        R=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        P=[fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
    )


def build_scene_update(t: float) -> schemas.SceneUpdate:
    """Build a SceneUpdate with animated cube, sphere, and lines."""
    yaw = t * 0.5
    return schemas.SceneUpdate(
        entities=[
            schemas.SceneEntity(
                timestamp=timestamp_now(),
                frame_id="world",
                id="demo_objects",
                cubes=[
                    schemas.CubePrimitive(
                        pose=schemas.Pose(
                            position=schemas.Vector3(x=1.5, y=0.0, z=0.5),
                            orientation=quat_from_yaw(yaw),
                        ),
                        size=schemas.Vector3(x=1.0, y=1.0, z=1.0),
                        color=schemas.Color(r=0.2, g=0.8, b=0.3, a=1.0),
                    )
                ],
                spheres=[
                    schemas.SpherePrimitive(
                        pose=schemas.Pose(
                            position=schemas.Vector3(x=-1.0, y=0.5, z=0.5),
                            orientation=schemas.Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                        ),
                        size=schemas.Vector3(x=0.6, y=0.6, z=0.6),
                        color=schemas.Color(r=0.3, g=0.4, b=0.9, a=0.9),
                    )
                ],
                lines=[
                    schemas.LinePrimitive(
                        type=schemas.LinePrimitiveLineType.LineStrip,
                        pose=schemas.Pose(
                            position=schemas.Vector3(x=0.0, y=0.0, z=0.0),
                            orientation=schemas.Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                        ),
                        thickness=0.05,
                        color=schemas.Color(r=1.0, g=0.6, b=0.2, a=1.0),
                        points=[
                            schemas.Point3(x=-2.0, y=0.0, z=0.0),
                            schemas.Point3(x=2.0, y=0.0, z=0.0),
                            schemas.Point3(x=2.0, y=2.0, z=0.0),
                        ],
                    )
                ],
            )
        ]
    )


def build_point_cloud(t: float) -> schemas.PointCloud:
    """Build a PointCloud with animated ring pattern."""
    import struct

    # PointCloud uses packed binary data
    # Each point needs: x, y, z, red, green, blue, alpha (all float32)
    data = bytearray()
    fields = [
        schemas.PackedElementField(
            name="x", offset=0, type=schemas.PackedElementFieldNumericType.Float32
        ),
        schemas.PackedElementField(
            name="y", offset=4, type=schemas.PackedElementFieldNumericType.Float32
        ),
        schemas.PackedElementField(
            name="z", offset=8, type=schemas.PackedElementFieldNumericType.Float32
        ),
        schemas.PackedElementField(
            name="red", offset=12, type=schemas.PackedElementFieldNumericType.Float32
        ),
        schemas.PackedElementField(
            name="green", offset=16, type=schemas.PackedElementFieldNumericType.Float32
        ),
        schemas.PackedElementField(
            name="blue", offset=20, type=schemas.PackedElementFieldNumericType.Float32
        ),
        schemas.PackedElementField(
            name="alpha", offset=24, type=schemas.PackedElementFieldNumericType.Float32
        ),
    ]
    point_stride = 28  # 7 fields * 4 bytes each

    for i in range(60):
        angle = (i / 60.0) * math.tau + t * 0.4
        radius = 1.0 + 0.2 * math.sin(t + i * 0.3)
        x = math.cos(angle) * radius
        y = math.sin(angle) * radius
        z = 0.3 * math.sin(t + i * 0.2)
        data.extend(struct.pack("<7f", x, y, z, 0.1, 0.8, 1.0, 1.0))

    return schemas.PointCloud(
        timestamp=timestamp_now(),
        frame_id="world",
        pose=schemas.Pose(
            position=schemas.Vector3(x=0.0, y=0.0, z=0.0),
            orientation=schemas.Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
        point_stride=point_stride,
        fields=fields,
        data=bytes(data),
    )


def build_frame_transforms(t: float) -> schemas.FrameTransforms:
    """Build FrameTransforms defining robot and camera poses."""
    return schemas.FrameTransforms(
        transforms=[
            # Animated: world -> robot
            schemas.FrameTransform(
                timestamp=timestamp_now(),
                parent_frame_id="world",
                child_frame_id="robot",
                translation=schemas.Vector3(
                    x=math.cos(t) * 0.5,
                    y=math.sin(t) * 0.5,
                    z=0.2,
                ),
                rotation=quat_from_yaw(t * 0.3),
            ),
            # Fixed: robot -> camera_link (mounted on front of robot)
            schemas.FrameTransform(
                timestamp=timestamp_now(),
                parent_frame_id="robot",
                child_frame_id="camera_link",
                translation=schemas.Vector3(x=0.3, y=0.0, z=0.1),
                rotation=quat_from_yaw(0.0),  # Facing forward
            ),
        ]
    )


def build_plot_sample(t: float) -> PlotSample:
    """Build a PlotSample message."""
    return PlotSample(
        t=t,
        sine=math.sin(t),
        cosine=math.cos(t),
        saw=(t % 6.0) - 3.0,
        noise=random.uniform(-0.2, 0.2),
    )


def build_custom_status(t: float) -> CustomStatus:
    """Build a CustomStatus message."""
    return CustomStatus(
        device_id="rig-01",
        state="RUNNING",
        battery=max(0.0, 1.0 - (t % 60.0) / 60.0),
        temperatures=[35.2 + math.sin(t) * 2.0, 36.1 + math.cos(t) * 1.5],
        events=[
            {
                "code": "heartbeat",
                "severity": "info",
                "message": f"tick={t:.2f}",
            }
        ],
    )


def build_log_message(tick: int) -> schemas.Log:
    """Build a Log message."""
    level = schemas.LogLevel.Info if tick % 5 else schemas.LogLevel.Warning
    return schemas.Log(
        timestamp=timestamp_now(),
        level=level,
        name="demo",
        message=f"publish tick {tick}",
    )


# =============================================================================
# Channel Setup
# =============================================================================


def setup_channels() -> Dict[str, foxglove.Channel]:
    """Set up all Foxglove channels with their schemas."""
    channels: Dict[str, foxglove.Channel] = {}

    # Channels with Pydantic-derived JSON schemas
    channels["plot"] = foxglove.Channel(
        "/plot/sample",
        schema=PlotSample.model_json_schema(),
    )
    channels["custom"] = foxglove.Channel(
        "/custom/status",
        schema=CustomStatus.model_json_schema(),
    )

    # Channels with typed Foxglove schema channels
    channels["tf"] = FrameTransformsChannel("/tf")
    channels["scene"] = SceneUpdateChannel("/viz/scene")
    channels["point_cloud"] = PointCloudChannel("/viz/point_cloud")
    channels["image"] = RawImageChannel("/viz/image")
    channels["camera_info"] = CameraCalibrationChannel("/viz/camera_info")
    channels["log"] = LogChannel("/system/log")

    return channels


# =============================================================================
# Main
# =============================================================================


def run_loop(channels: Dict[str, foxglove.Channel], rate_hz: float) -> None:
    """Main publish loop."""
    start = time.time()
    tick = 0
    period = 1.0 / max(1e-3, rate_hz)

    try:
        while True:
            t = time.time() - start
            timestamp_ns = time.time_ns()

            # Publish plot sample (Pydantic model)
            plot_msg = build_plot_sample(t)
            channels["plot"].log(plot_msg.model_dump(), log_time=timestamp_ns)

            # Publish custom status (Pydantic model)
            custom_msg = build_custom_status(t)
            channels["custom"].log(custom_msg.model_dump(), log_time=timestamp_ns)

            # Publish frame transforms (for 3D coordinate frames)
            transforms_msg = build_frame_transforms(t)
            channels["tf"].log(transforms_msg, log_time=timestamp_ns)

            # Publish scene update (3D primitives)
            scene_msg = build_scene_update(t)
            channels["scene"].log(scene_msg, log_time=timestamp_ns)

            # Publish point cloud
            point_cloud_msg = build_point_cloud(t)
            channels["point_cloud"].log(point_cloud_msg, log_time=timestamp_ns)

            # Publish image
            image_msg = encode_image_rgb(160, 120, t)
            channels["image"].log(image_msg, log_time=timestamp_ns)

            # Publish camera calibration (matches the image)
            camera_info_msg = build_camera_calibration(160, 120)
            channels["camera_info"].log(camera_info_msg, log_time=timestamp_ns)

            # Publish log message
            log_msg = build_log_message(tick)
            channels["log"].log(log_msg, log_time=timestamp_ns)

            tick += 1
            time.sleep(period)

    except KeyboardInterrupt:
        print("\nShutting down...")


def run(host: str, port: int, rate_hz: float, mcap_path: str | None) -> None:
    # Start the WebSocket server
    server = foxglove.start_server(host=host, port=port, name="foxglove-viz-demo")
    print(f"Foxglove server started at ws://{host}:{port}")

    # Create all channels (must be created before MCAP writer to be recorded)
    channels = setup_channels()

    if mcap_path:
        # Use context manager for MCAP recording (auto-closes on exit)
        with foxglove.open_mcap(mcap_path, allow_overwrite=True) as mcap_writer:
            print(f"Recording to {mcap_path}")
            run_loop(channels, rate_hz)
        print(f"MCAP recording saved to {mcap_path}")
    else:
        run_loop(channels, rate_hz)

    server.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Foxglove visualization demo server using pydantic schemas"
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8765, help="WebSocket port (default: 8765)"
    )
    parser.add_argument(
        "--rate", type=float, default=10.0, help="Publish rate in Hz (default: 10)"
    )
    parser.add_argument(
        "--mcap",
        default="foxglove-demo.mcap",
        help="MCAP output path (default: foxglove-demo.mcap, use --no-mcap to disable)",
    )
    parser.add_argument(
        "--no-mcap",
        action="store_true",
        help="Disable MCAP recording",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mcap_path = None if args.no_mcap else args.mcap
    run(args.host, args.port, args.rate, mcap_path)


if __name__ == "__main__":
    main()
