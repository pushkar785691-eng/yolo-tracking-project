"""
track_and_count.py

Real-time multi-object detection + tracking + line-crossing counting,
built on YOLOv8 (Ultralytics) + ByteTrack.

Usage:
    python src/track_and_count.py --source 0                     # webcam
    python src/track_and_count.py --source sample/test_clip.mp4   # video file
    python src/track_and_count.py --source path/to.mp4 --config config.yaml

Press 'q' to quit an on-screen preview window (when --show is used).
"""

import argparse
import csv
import os
import time

import cv2
import yaml
from ultralytics import YOLO

from line_counter import LineZone, TrackHistory


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 + ByteTrack detection, tracking and line counting.")
    parser.add_argument("--source", type=str, default="sample/sample_video.mp4",
                         help="Video file path, or '0' for the default webcam.")
    parser.add_argument("--config", type=str, default="config.yaml",
                         help="Path to the YAML config file.")
    parser.add_argument("--show", action="store_true",
                         help="Show a live preview window while processing.")
    parser.add_argument("--max-frames", type=int, default=None,
                         help="Optional cap on number of frames to process (useful for quick tests).")
    return parser.parse_args()


def resolve_source(source: str):
    # Allow '0', '1', etc. to mean webcam index instead of a file path.
    if source.isdigit():
        return int(source)
    return source


def draw_dashboard(frame, line_zone: LineZone, fps: float):
    """Draws the counting line and a small stats panel onto the frame."""
    a, b = line_zone.point_a, line_zone.point_b
    cv2.line(frame, a, b, (0, 255, 255), 2)

    panel_h = 90
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, panel_h), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    cv2.putText(frame, f"IN:  {line_zone.in_count}", (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"OUT: {line_zone.out_count}", (12, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"NET: {line_zone.net_count()}   FPS: {fps:.1f}", (12, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


def main():
    args = parse_args()
    cfg = load_config(args.config)

    model = YOLO(cfg["model"]["weights"])
    class_names = model.names

    line_cfg = cfg["counting_line"]
    line_zone = LineZone(
        point_a=tuple(line_cfg["point_a"]),
        point_b=tuple(line_cfg["point_b"]),
        name=line_cfg.get("name", "line"),
    )
    history = TrackHistory(max_len=cfg["output"].get("trail_length", 30))

    source = resolve_source(args.source)

    writer = None
    out_path = cfg["output"].get("output_path", "outputs/annotated_output.mp4")
    if cfg["output"].get("save_video", True):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    crossing_log_path = os.path.join(os.path.dirname(out_path) or ".", "crossing_log.csv")
    crossing_rows = []

    frame_idx = 0
    t_start = time.time()

    stream = model.track(
        source=source,
        tracker=cfg["model"]["tracker"],
        conf=cfg["model"]["confidence"],
        classes=cfg["model"]["classes"],
        persist=True,
        stream=True,
        verbose=False,
    )

    for result in stream:
        frame = result.orig_img.copy()
        h, w = frame.shape[:2]

        if writer is None and cfg["output"].get("save_video", True):
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps_in = result.speed.get("fps", 20) if hasattr(result, "speed") else 20
            writer = cv2.VideoWriter(out_path, fourcc, 20.0, (w, h))

        active_ids = []
        boxes = result.boxes

        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int)
            clss = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()

            for box, track_id, cls_id, conf in zip(xyxy, ids, clss, confs):
                x1, y1, x2, y2 = box.astype(int)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                active_ids.append(track_id)

                # update trail
                trail = history.update(track_id, (cx, cy))

                # update line-crossing state
                direction = line_zone.update(track_id, (cx, cy), frame_idx)
                if direction:
                    crossing_rows.append({
                        "frame": frame_idx,
                        "track_id": track_id,
                        "class": class_names.get(cls_id, str(cls_id)),
                        "direction": direction,
                    })

                # draw box + label
                label = f"#{track_id} {class_names.get(cls_id, cls_id)} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 220, 60), 2)
                cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 220, 60), 1)

                # draw motion trail
                if cfg["output"].get("draw_trails", True) and len(trail) > 1:
                    for i in range(1, len(trail)):
                        cv2.line(frame, trail[i - 1], trail[i], (200, 200, 0), 2)

        history.prune(active_ids)

        elapsed = time.time() - t_start
        fps = frame_idx / elapsed if elapsed > 0 else 0.0
        frame = draw_dashboard(frame, line_zone, fps)

        if writer is not None:
            writer.write(frame)

        if args.show:
            cv2.imshow("YOLOv8 + ByteTrack | Detection, Tracking & Counting", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if args.max_frames and frame_idx >= args.max_frames:
            break

    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    # write crossing log
    if crossing_rows:
        with open(crossing_log_path, "w", newline="") as f:
            writer_csv = csv.DictWriter(f, fieldnames=["frame", "track_id", "class", "direction"])
            writer_csv.writeheader()
            writer_csv.writerows(crossing_rows)

    print("\n===== Run summary =====")
    print(f"Frames processed : {frame_idx}")
    print(f"IN count         : {line_zone.in_count}")
    print(f"OUT count        : {line_zone.out_count}")
    print(f"NET count        : {line_zone.net_count()}")
    if cfg["output"].get("save_video", True):
        print(f"Annotated video  : {out_path}")
    if crossing_rows:
        print(f"Crossing log     : {crossing_log_path}")


if __name__ == "__main__":
    main()
