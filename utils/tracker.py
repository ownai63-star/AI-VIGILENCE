from deep_sort_realtime.deepsort_tracker import DeepSort

class ObjectTracker:
    def __init__(self, max_age=60, n_init=1):
        # n_init=1: confirm track on first detection (most responsive for live video)
        # max_age=60: keep track alive for 60 missed frames (handles brief occlusion)
        self.tracker = DeepSort(max_age=max_age, n_init=n_init)

    def update(self, detections, frame):
        """
        detections: list of ([x1, y1, w, h], confidence, label)
        """
        tracks = self.tracker.update_tracks(detections, frame=frame)
        active_tracks = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()  # [left, top, right, bottom]
            active_tracks.append({
                'id': track_id,
                'bbox': [float(ltrb[0]), float(ltrb[1]), float(ltrb[2]), float(ltrb[3])]
            })
        return active_tracks
