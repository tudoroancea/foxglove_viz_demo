import argparse
import asyncio
import base64
import json
import math
import random
import time
from typing import Any, Dict, List

from foxglove_websocket.server import FoxgloveServer


def now_sec_nsec() -> Dict[str, int]:
    ts = time.time_ns()
    return {"sec": ts // 1_000_000_000, "nsec": ts % 1_000_000_000}


def quat_from_yaw(yaw: float) -> Dict[str, float]:
    half = yaw * 0.5
    return {"x": 0.0, "y": 0.0, "z": math.sin(half), "w": math.cos(half)}


def encode_image_rgb(width: int, height: int, t: float) -> Dict[str, Any]:
    data = bytearray()
    for y in range(height):
        for x in range(width):
            r = int((x / max(1, width - 1)) * 255)
            g = int((y / max(1, height - 1)) * 255)
            b = int(((math.sin(t + x * 0.1) + 1.0) * 0.5) * 255)
            data.extend((r, g, b))
    return {
        "timestamp": now_sec_nsec(),
        "frame_id": "camera",
        "height": height,
        "width": width,
        "encoding": "rgb8",
        "step": width * 3,
        "data": base64.b64encode(data).decode("ascii"),
    }


def build_scene_update(t: float) -> Dict[str, Any]:
    yaw = t * 0.5
    return {
        "deletes": [],
        "entities": [
            {
                "id": "cube",
                "frame_id": "world",
                "lifetime": {"sec": 0, "nsec": 0},
                "metadata": [{"key": "label", "value": "Spinning cube"}],
                "cubes": [
                    {
                        "pose": {
                            "position": {"x": 1.5, "y": 0.0, "z": 0.5},
                            "orientation": quat_from_yaw(yaw),
                        },
                        "size": {"x": 1.0, "y": 1.0, "z": 1.0},
                        "color": {"r": 0.2, "g": 0.8, "b": 0.3, "a": 1.0},
                    }
                ],
                "spheres": [
                    {
                        "pose": {
                            "position": {"x": -1.0, "y": 0.5, "z": 0.5},
                            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                        },
                        "size": {"x": 0.6, "y": 0.6, "z": 0.6},
                        "color": {"r": 0.3, "g": 0.4, "b": 0.9, "a": 0.9},
                    }
                ],
                "lines": [
                    {
                        "pose": {
                            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                        },
                        "thickness": 0.05,
                        "color": {"r": 1.0, "g": 0.6, "b": 0.2, "a": 1.0},
                        "points": [
                            {"x": -2.0, "y": 0.0, "z": 0.0},
                            {"x": 2.0, "y": 0.0, "z": 0.0},
                            {"x": 2.0, "y": 2.0, "z": 0.0},
                        ],
                    }
                ],
            }
        ],
    }


def build_point_cloud(t: float) -> Dict[str, Any]:
    points: List[Dict[str, Any]] = []
    for i in range(60):
        angle = (i / 60.0) * math.tau + t * 0.4
        radius = 1.0 + 0.2 * math.sin(t + i * 0.3)
        points.append(
            {
                "x": math.cos(angle) * radius,
                "y": math.sin(angle) * radius,
                "z": 0.3 * math.sin(t + i * 0.2),
                "color": {"r": 0.1, "g": 0.8, "b": 1.0, "a": 1.0},
            }
        )
    return {
        "timestamp": now_sec_nsec(),
        "frame_id": "world",
        "points": points,
    }


def build_pose(t: float) -> Dict[str, Any]:
    return {
        "position": {"x": math.cos(t), "y": math.sin(t), "z": 0.2},
        "orientation": quat_from_yaw(t),
    }


def build_transform(t: float) -> Dict[str, Any]:
    return {
        "timestamp": now_sec_nsec(),
        "parent_frame_id": "world",
        "child_frame_id": "robot",
        "translation": {"x": math.cos(t) * 0.5, "y": math.sin(t) * 0.5, "z": 0.2},
        "rotation": quat_from_yaw(t * 0.3),
    }


def build_plot_sample(t: float) -> Dict[str, Any]:
    return {
        "t": t,
        "sine": math.sin(t),
        "cosine": math.cos(t),
        "saw": (t % 6.0) - 3.0,
        "noise": random.uniform(-0.2, 0.2),
    }


def build_custom_status(t: float) -> Dict[str, Any]:
    return {
        "device_id": "rig-01",
        "state": "RUNNING",
        "battery": max(0.0, 1.0 - (t % 60.0) / 60.0),
        "temperatures": [35.2 + math.sin(t) * 2.0, 36.1 + math.cos(t) * 1.5],
        "events": [
            {
                "code": "heartbeat",
                "severity": "info",
                "message": f"tick={t:.2f}",
            }
        ],
    }


PLOT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "demo.PlotSample",
    "type": "object",
    "properties": {
        "t": {"type": "number"},
        "sine": {"type": "number"},
        "cosine": {"type": "number"},
        "saw": {"type": "number"},
        "noise": {"type": "number"},
    },
    "required": ["t", "sine", "cosine", "saw", "noise"],
}

CUSTOM_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "demo.CustomStatus",
    "type": "object",
    "properties": {
        "device_id": {"type": "string"},
        "state": {"type": "string"},
        "battery": {"type": "number"},
        "temperatures": {"type": "array", "items": {"type": "number"}},
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "severity": {"type": "string"},
                    "message": {"type": "string"},
                },
                "required": ["code", "severity", "message"],
            },
        },
    },
    "required": ["device_id", "state", "battery", "temperatures", "events"],
}


async def maybe_await(result: Any) -> Any:
    if asyncio.iscoroutine(result):
        return await result
    return result


async def run(host: str, port: int, rate_hz: float) -> None:
    server = FoxgloveServer(host, port, "foxglove-viz-demo")
    await maybe_await(server.start())

    channels: Dict[str, int] = {}
    channels["plot"] = await maybe_await(
        server.add_channel(
            {
                "topic": "plot/sample",
                "encoding": "json",
                "schemaName": "demo.PlotSample",
                "schema": json.dumps(PLOT_SCHEMA),
                "schemaEncoding": "jsonschema",
            }
        )
    )
    channels["custom"] = await maybe_await(
        server.add_channel(
            {
                "topic": "custom/status",
                "encoding": "json",
                "schemaName": "demo.CustomStatus",
                "schema": json.dumps(CUSTOM_SCHEMA),
                "schemaEncoding": "jsonschema",
            }
        )
    )
    channels["pose"] = await maybe_await(
        server.add_channel(
            {
                "topic": "viz/pose",
                "encoding": "json",
                "schemaName": "foxglove.Pose",
                "schema": "",
            }
        )
    )
    channels["transform"] = await maybe_await(
        server.add_channel(
            {
                "topic": "viz/transform",
                "encoding": "json",
                "schemaName": "foxglove.FrameTransform",
                "schema": "",
            }
        )
    )
    channels["scene"] = await maybe_await(
        server.add_channel(
            {
                "topic": "viz/scene",
                "encoding": "json",
                "schemaName": "foxglove.SceneUpdate",
                "schema": "",
            }
        )
    )
    channels["point_cloud"] = await maybe_await(
        server.add_channel(
            {
                "topic": "viz/point_cloud",
                "encoding": "json",
                "schemaName": "foxglove.PointCloud",
                "schema": "",
            }
        )
    )
    channels["image"] = await maybe_await(
        server.add_channel(
            {
                "topic": "viz/image",
                "encoding": "json",
                "schemaName": "foxglove.Image",
                "schema": "",
            }
        )
    )
    channels["log"] = await maybe_await(
        server.add_channel(
            {
                "topic": "system/log",
                "encoding": "json",
                "schemaName": "foxglove.Log",
                "schema": "",
            }
        )
    )

    start = time.time()
    tick = 0
    period = 1.0 / max(1e-3, rate_hz)

    try:
        while True:
            t = time.time() - start
            timestamp_ns = time.time_ns()

            plot_msg = build_plot_sample(t)
            await maybe_await(
                server.send_message(
                    channels["plot"], timestamp_ns, json.dumps(plot_msg).encode()
                )
            )

            custom_msg = build_custom_status(t)
            await maybe_await(
                server.send_message(
                    channels["custom"], timestamp_ns, json.dumps(custom_msg).encode()
                )
            )

            pose_msg = build_pose(t)
            await maybe_await(
                server.send_message(
                    channels["pose"], timestamp_ns, json.dumps(pose_msg).encode()
                )
            )

            transform_msg = build_transform(t)
            await maybe_await(
                server.send_message(
                    channels["transform"],
                    timestamp_ns,
                    json.dumps(transform_msg).encode(),
                )
            )

            scene_msg = build_scene_update(t)
            await maybe_await(
                server.send_message(
                    channels["scene"], timestamp_ns, json.dumps(scene_msg).encode()
                )
            )

            point_cloud_msg = build_point_cloud(t)
            await maybe_await(
                server.send_message(
                    channels["point_cloud"],
                    timestamp_ns,
                    json.dumps(point_cloud_msg).encode(),
                )
            )

            image_msg = encode_image_rgb(160, 120, t)
            await maybe_await(
                server.send_message(
                    channels["image"], timestamp_ns, json.dumps(image_msg).encode()
                )
            )

            log_msg = {
                "timestamp": now_sec_nsec(),
                "level": "info" if tick % 5 else "warning",
                "name": "demo",
                "message": f"publish tick {tick}",
            }
            await maybe_await(
                server.send_message(
                    channels["log"], timestamp_ns, json.dumps(log_msg).encode()
                )
            )

            tick += 1
            await asyncio.sleep(period)
    finally:
        stop = getattr(server, "stop", None) or getattr(server, "close", None)
        if stop is not None:
            await maybe_await(stop())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Foxglove visualization demo server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port")
    parser.add_argument("--rate", type=float, default=10.0, help="Publish rate (Hz)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(run(args.host, args.port, args.rate))


if __name__ == "__main__":
    main()
