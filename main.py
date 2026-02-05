import argparse
import math
import random
import time
from dataclasses import dataclass
from typing import Any

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
    temperatures: list[float]
    events: list[dict]  # Use dict to avoid $ref in JSON schema


# =============================================================================
# Utility Functions
# =============================================================================


def now_sec_nsec() -> dict[str, int]:
    ts = time.time_ns()
    return {"sec": ts // 1_000_000_000, "nsec": ts % 1_000_000_000}


def timestamp_from_seconds(seconds: float) -> schemas.Timestamp:
    """Convert seconds to Foxglove Timestamp."""
    sec = int(seconds)
    nsec = int((seconds - sec) * 1_000_000_000)
    return schemas.Timestamp(sec=sec, nsec=nsec)


def timestamp_now() -> schemas.Timestamp:
    """Get current timestamp as Foxglove Timestamp."""
    ts = time.time_ns()
    return schemas.Timestamp(sec=ts // 1_000_000_000, nsec=ts % 1_000_000_000)


def quat_from_yaw(yaw: float) -> schemas.Quaternion:
    """Convert yaw angle to Quaternion."""
    half = yaw * 0.5
    return schemas.Quaternion(
        x=0.0,
        y=0.0,
        z=math.sin(half),
        w=math.cos(half),
    )


# =============================================================================
# Message Builders
# =============================================================================


def encode_image_rgb(
    width: int,
    height: int,
    t: float,
    use_sim_time: bool = False,
    sim_time_offset: float = 0.0,
) -> schemas.RawImage:
    """Build a RawImage with gradient pattern."""
    data = bytearray()
    for y in range(height):
        for x in range(width):
            r = int((x / max(1, width - 1)) * 255)
            g = int((y / max(1, height - 1)) * 255)
            b = int(((math.sin(t + x * 0.1) + 1.0) * 0.5) * 255)
            data.extend((r, g, b))
    timestamp = timestamp_from_seconds(t) if use_sim_time else timestamp_now()
    return schemas.RawImage(
        timestamp=timestamp,
        frame_id="camera",
        height=height,
        width=width,
        encoding="rgb8",
        step=width * 3,
        data=bytes(data),
    )


def build_camera_calibration(
    width: int, height: int, use_sim_time: bool = False, sim_time: float = 0.0
) -> schemas.CameraCalibration:
    """Build CameraCalibration for the camera."""
    # Simple pinhole camera model
    fx = fy = max(width, height)  # focal length in pixels
    cx = width / 2.0  # principal point x
    cy = height / 2.0  # principal point y
    timestamp = timestamp_from_seconds(sim_time) if use_sim_time else timestamp_now()
    return schemas.CameraCalibration(
        timestamp=timestamp,
        frame_id="camera_link",
        height=height,
        width=width,
        distortion_model="plumb_bob",
        D=[0.0, 0.0, 0.0, 0.0, 0.0],  # No distortion
        K=[fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        R=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        P=[fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
    )


def build_scene_update(t: float, use_sim_time: bool = False) -> schemas.SceneUpdate:
    """Build a SceneUpdate with animated cube, sphere, and lines."""
    yaw = t * 0.5
    timestamp = timestamp_from_seconds(t) if use_sim_time else timestamp_now()
    return schemas.SceneUpdate(
        entities=[
            schemas.SceneEntity(
                timestamp=timestamp,
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


def build_point_cloud(t: float, use_sim_time: bool = False) -> schemas.PointCloud:
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

    timestamp = timestamp_from_seconds(t) if use_sim_time else timestamp_now()
    return schemas.PointCloud(
        timestamp=timestamp,
        frame_id="world",
        pose=schemas.Pose(
            position=schemas.Vector3(x=0.0, y=0.0, z=0.0),
            orientation=schemas.Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
        point_stride=point_stride,
        fields=fields,
        data=bytes(data),
    )


def build_frame_transforms(
    t: float, use_sim_time: bool = False
) -> schemas.FrameTransforms:
    """Build FrameTransforms defining robot and camera poses."""
    timestamp = timestamp_from_seconds(t) if use_sim_time else timestamp_now()
    return schemas.FrameTransforms(
        transforms=[
            # Animated: world -> robot
            schemas.FrameTransform(
                timestamp=timestamp,
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
                timestamp=timestamp,
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


def build_log_message(tick: int, t: float, use_sim_time: bool = False) -> schemas.Log:
    """Build a Log message."""
    level = schemas.LogLevel.Info if tick % 5 else schemas.LogLevel.Warning
    timestamp = timestamp_from_seconds(t) if use_sim_time else timestamp_now()
    return schemas.Log(
        timestamp=timestamp,
        level=level,
        name="demo",
        message=f"publish tick {tick}",
    )


# =============================================================================
# Channel Setup
# =============================================================================


def setup_channels() -> dict[str, foxglove.Channel]:
    """Set up all Foxglove channels with their schemas."""
    channels: dict[str, foxglove.Channel] = {}

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
# Message Generation Helpers
# =============================================================================


@dataclass
class MessageBatch:
    """Container for all messages at a single timestep."""

    t: float
    timestamp_ns: int
    plot: PlotSample
    custom: CustomStatus
    transforms: schemas.FrameTransforms
    scene: schemas.SceneUpdate
    point_cloud: schemas.PointCloud
    image: schemas.RawImage
    camera_info: schemas.CameraCalibration
    log: schemas.Log


def generate_message_batch(
    t: float, tick: int, use_sim_time: bool = False
) -> MessageBatch:
    """Generate all messages for a single timestep."""
    timestamp_ns = int(t * 1_000_000_000) if use_sim_time else time.time_ns()

    return MessageBatch(
        t=t,
        timestamp_ns=timestamp_ns,
        plot=build_plot_sample(t),
        custom=build_custom_status(t),
        transforms=build_frame_transforms(t, use_sim_time),
        scene=build_scene_update(t, use_sim_time),
        point_cloud=build_point_cloud(t, use_sim_time),
        image=encode_image_rgb(160, 120, t, use_sim_time),
        camera_info=build_camera_calibration(160, 120, use_sim_time, t),
        log=build_log_message(tick, t, use_sim_time),
    )


def publish_batch(channels: dict[str, foxglove.Channel], batch: MessageBatch) -> None:
    """Publish a batch of messages to all channels."""
    channels["plot"].log(batch.plot.model_dump(), log_time=batch.timestamp_ns)
    channels["custom"].log(batch.custom.model_dump(), log_time=batch.timestamp_ns)
    channels["tf"].log(batch.transforms, log_time=batch.timestamp_ns)
    channels["scene"].log(batch.scene, log_time=batch.timestamp_ns)
    channels["point_cloud"].log(batch.point_cloud, log_time=batch.timestamp_ns)
    channels["image"].log(batch.image, log_time=batch.timestamp_ns)
    channels["camera_info"].log(batch.camera_info, log_time=batch.timestamp_ns)
    channels["log"].log(batch.log, log_time=batch.timestamp_ns)


# =============================================================================
# Mode Implementations
# =============================================================================


def run_live_mode(
    channels: dict[str, foxglove.Channel], rate_hz: float, duration: float | None
) -> None:
    """Run live visualization mode - publishes messages as they are created."""
    start = time.time()
    tick = 0
    period = 1.0 / max(1e-3, rate_hz)

    try:
        while True:
            t = time.time() - start

            if duration is not None and t >= duration:
                print(f"\nReached duration limit of {duration}s")
                break

            batch = generate_message_batch(t, tick, use_sim_time=False)
            publish_batch(channels, batch)

            tick += 1
            time.sleep(period)

    except KeyboardInterrupt:
        print("\nShutting down...")


def run_batch_mode(
    channels: dict[str, foxglove.Channel], rate_hz: float, duration: float
) -> None:
    """Run batch mode - compute all messages first, then publish all at once with sim timestamps."""
    print(f"Computing {duration}s of data at {rate_hz}Hz...")

    period = 1.0 / max(1e-3, rate_hz)
    num_frames = int(duration * rate_hz)

    # Pre-compute all messages
    batches: list[MessageBatch] = []
    for tick in range(num_frames):
        t = tick * period
        batch = generate_message_batch(t, tick, use_sim_time=True)
        batches.append(batch)

    print(f"Computed {len(batches)} message batches")
    print(f"Publishing all messages now...")

    # Publish all messages at once (WebSocket can handle rapid publishes)
    for batch in batches:
        publish_batch(channels, batch)

    print(f"Published all {len(batches)} batches")


def run_file_mode(mcap_path: str, rate_hz: float, duration: float) -> None:
    """Run file mode - record to MCAP file with artificial timestamps (fast generation)."""
    if duration is None:
        print("Error: --duration is required for file mode")
        return

    # Create all channels (must be created before MCAP writer to be recorded)
    channels = setup_channels()

    with foxglove.open_mcap(mcap_path, allow_overwrite=True) as mcap_writer:
        print(f"Generating {duration}s of data at {rate_hz}Hz...")

        period = 1.0 / max(1e-3, rate_hz)
        num_frames = int(duration * rate_hz)

        # Generate and write all messages as fast as possible with artificial timestamps
        for tick in range(num_frames):
            t = tick * period
            batch = generate_message_batch(t, tick, use_sim_time=True)
            publish_batch(channels, batch)

            if (tick + 1) % 100 == 0 or tick == num_frames - 1:
                print(f"  Generated {tick + 1}/{num_frames} frames...", end="\r")

        print(f"\nGenerated {num_frames} frames")

    print(f"MCAP recording saved to {mcap_path}")


# =============================================================================
# Main
# =============================================================================


def run(
    mode: str,
    host: str,
    port: int,
    rate_hz: float,
    duration: float | None,
    mcap_path: str | None,
) -> None:
    if mode == "live":
        # Start the WebSocket server
        server = foxglove.start_server(host=host, port=port, name="foxglove-viz-demo")
        print(f"Foxglove server started at ws://{host}:{port}")

        channels = setup_channels()
        run_live_mode(channels, rate_hz, duration)
        server.stop()

    elif mode == "batch":
        if duration is None:
            print("Error: --duration is required for batch mode")
            return

        # Start the WebSocket server
        server = foxglove.start_server(host=host, port=port, name="foxglove-viz-demo")
        print(f"Foxglove server started at ws://{host}:{port}")

        channels = setup_channels()
        run_batch_mode(channels, rate_hz, duration)

        # Keep server running so user can view the data
        print("\nData published. Press Ctrl+C to stop the server.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
        server.stop()

    elif mode == "file":
        if mcap_path is None:
            print("Error: --mcap is required for file mode")
            return
        if duration is None:
            print("Error: --duration is required for file mode")
            return
        run_file_mode(mcap_path, rate_hz, duration)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Foxglove visualization demo server")

    # Mode selection - mutually exclusive
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--live",
        action="store_true",
        help="Live visualization mode - publish messages as they are created",
    )
    mode_group.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode - compute all messages first, then publish with sim timestamps",
    )
    mode_group.add_argument(
        "--file",
        action="store_true",
        help="File mode - record to MCAP file",
    )

    # Common options
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
        "--duration",
        type=float,
        default=None,
        help="Duration in seconds (required for batch mode, optional for others)",
    )
    parser.add_argument(
        "--mcap",
        default="foxglove-demo.mcap",
        help="MCAP output path for file mode (default: foxglove-demo.mcap)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Determine mode
    if args.live:
        mode = "live"
    elif args.batch:
        mode = "batch"
    else:
        mode = "file"

    run(
        mode=mode,
        host=args.host,
        port=args.port,
        rate_hz=args.rate,
        duration=args.duration,
        mcap_path=args.mcap if mode == "file" else None,
    )


if __name__ == "__main__":
    main()
