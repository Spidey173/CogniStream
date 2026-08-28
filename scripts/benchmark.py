#!/usr/bin/env python3
"""
End-to-End Performance Benchmarking Suite.
Simulates high-FPS multi-camera WebSocket frame ingestion and measures latency, FPS, and throughput.
Exports a detailed JSON performance report.
"""

import argparse
import asyncio
import io
import json
import time
from datetime import datetime, timezone
import numpy as np
from PIL import Image
import websockets


def generate_synthetic_jpeg(width: int = 640, height: int = 480) -> bytes:
    """Generate synthetic test frame image bytes."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = 100
    frame[:, :, 1] = 150
    frame[:, :, 2] = 200
    img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return buf.getvalue()


async def benchmark_client(url: str, camera_id: str, fps: int, duration: float) -> dict:
    """Simulate a single camera streaming frames over WebSocket."""
    jpeg_bytes = generate_synthetic_jpeg()
    interval = 1.0 / fps
    total_frames = int(duration * fps)

    latencies = []
    start_time = time.perf_counter()
    sent_count = 0
    ack_count = 0

    try:
        async with websockets.connect(f"{url}/{camera_id}?api_key=dev-secret-api-key") as ws:
            for _ in range(total_frames):
                frame_start = time.perf_counter()
                await ws.send(jpeg_bytes)
                sent_count += 1

                ack = await ws.recv()
                ack_data = json.loads(ack)
                if ack_data.get("status") == "ok":
                    ack_count += 1
                    latencies.append((time.perf_counter() - frame_start) * 1000.0)

                elapsed = time.perf_counter() - frame_start
                sleep_time = max(0.0, interval - elapsed)
                await asyncio.sleep(sleep_time)

    except Exception as e:
        print(f"[{camera_id}] Error in benchmark client: {e}")

    total_duration = time.perf_counter() - start_time
    actual_fps = ack_count / total_duration if total_duration > 0 else 0.0

    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    p95_lat = np.percentile(latencies, 95) if latencies else 0.0

    return {
        "camera_id": camera_id,
        "target_fps": fps,
        "actual_fps": round(actual_fps, 2),
        "sent_frames": sent_count,
        "received_acks": ack_count,
        "avg_latency_ms": round(avg_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "duration_seconds": round(total_duration, 2),
    }


async def run_benchmark(url: str, cameras: int, fps: int, duration: float, output: str):
    """Run concurrent benchmark clients across multiple cameras."""
    print(f"=== Starting Benchmark Suite ===")
    print(f"Target URL:       {url}")
    print(f"Cameras Count:    {cameras}")
    print(f"Target FPS:       {fps} FPS")
    print(f"Duration:         {duration} seconds")
    print(f"---------------------------------")

    tasks = [
        benchmark_client(url, f"cam_{i+1}", fps, duration)
        for i in range(cameras)
    ]

    results = await asyncio.gather(*tasks)

    total_acks = sum(r["received_acks"] for r in results)
    avg_fps = sum(r["actual_fps"] for r in results)
    avg_lat = sum(r["avg_latency_ms"] for r in results) / len(results) if results else 0.0

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_cameras": cameras,
        "aggregate_fps": round(avg_fps, 2),
        "average_latency_ms": round(avg_lat, 2),
        "total_frames_processed": total_acks,
        "camera_results": results,
    }

    print("\n=== BENCHMARK RESULTS SUMMARY ===")
    print(f"Aggregate System Throughput:  {summary['aggregate_fps']} FPS")
    print(f"Average Frame Latency:        {summary['average_latency_ms']} ms")
    print(f"Total Processed Frames:       {summary['total_frames_processed']}")
    print("=================================\n")

    if output:
        with open(output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Exported benchmark report to: {output}")


def main():
    parser = argparse.ArgumentParser(description="Real-Time Vision Stream Benchmark Tool")
    parser.add_argument("--url", default="ws://localhost:8000/api/v1/stream", help="WebSocket ingest base URL")
    parser.add_argument("--cameras", type=int, default=1, help="Number of concurrent camera streams")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS per camera")
    parser.add_argument("--duration", type=float, default=5.0, help="Benchmark duration in seconds")
    parser.add_argument("--output", default="benchmark_report.json", help="Output JSON report filename")
    args = parser.parse_args()

    asyncio.run(run_benchmark(args.url, args.cameras, args.fps, args.duration, args.output))


if __name__ == "__main__":
    main()
