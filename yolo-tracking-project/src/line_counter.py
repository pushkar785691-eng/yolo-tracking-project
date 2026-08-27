"""
line_counter.py

Implements a virtual "trip-wire" line that counts tracked objects as they
cross it, and reports which direction they crossed in.

The math: for a line defined by two points A and B, any point P lies on one
side or the other. We compute this using the sign of the 2D cross product
of (B - A) and (P - A). If an object's center point was on one side in the
previous frame and on the other side in the current frame, it crossed the
line. The sign of the change tells us the direction (e.g. "in" vs "out").
"""

from collections import defaultdict
from dataclasses import dataclass, field


def _side_of_line(a, b, p):
    """
    Returns a signed value indicating which side of line AB the point P is on.
    Positive -> one side, Negative -> the other side, ~0 -> on the line.
    """
    ax, ay = a
    bx, by = b
    px, py = p
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


@dataclass
class LineZone:
    """A single counting line between point_a and point_b."""

    point_a: tuple
    point_b: tuple
    name: str = "line"

    # internal state
    _last_side: dict = field(default_factory=dict)
    in_count: int = 0
    out_count: int = 0
    crossings: list = field(default_factory=list)  # log of (track_id, direction, frame_idx)

    def update(self, track_id: int, center: tuple, frame_idx: int = -1):
        """
        Call this once per frame per tracked object with its current center
        point. Returns "in", "out", or None if no crossing happened this call.
        """
        side = _side_of_line(self.point_a, self.point_b, center)
        prev_side = self._last_side.get(track_id)

        direction = None
        if prev_side is not None and prev_side != 0 and side != 0:
            if (prev_side > 0) and (side < 0):
                direction = "in"
                self.in_count += 1
            elif (prev_side < 0) and (side > 0):
                direction = "out"
                self.out_count += 1

            if direction:
                self.crossings.append((track_id, direction, frame_idx))

        self._last_side[track_id] = side
        return direction

    def net_count(self):
        return self.in_count - self.out_count

    def forget(self, track_id: int):
        """Drop a track that is no longer being tracked (e.g. left the frame)."""
        self._last_side.pop(track_id, None)


class TrackHistory:
    """
    Keeps a short trail of recent center points per track id, used both for
    drawing motion trails and for smoothing direction estimates.
    """

    def __init__(self, max_len: int = 30):
        self.max_len = max_len
        self._history = defaultdict(list)

    def update(self, track_id: int, center: tuple):
        pts = self._history[track_id]
        pts.append(center)
        if len(pts) > self.max_len:
            pts.pop(0)
        return pts

    def get(self, track_id: int):
        return self._history.get(track_id, [])

    def active_ids(self):
        return list(self._history.keys())

    def prune(self, active_ids):
        """Remove history for track ids that are no longer present."""
        stale = [tid for tid in self._history if tid not in active_ids]
        for tid in stale:
            del self._history[tid]
