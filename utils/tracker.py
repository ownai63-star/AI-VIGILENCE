from deep_sort_realtime.deepsort_tracker import DeepSort

class ObjectTracker:
    def __init__(self, max_age=15, n_init=2):
        self.tracker = DeepSort(max_age=max_age, n_init=n_init)
        # Map track_id -> last known tight bbox [x1,y1,x2,y2] from actual detection
        self._det_bbox: dict = {}

    def update(self, detections, frame):
        """
        detections: list of ([x1, y1, w, h], confidence, label)
        Returns tracks using the original detection bbox when available,
        so the box stays tight around the actual person.
        """
        tracks = self.tracker.update_tracks(detections, frame=frame)

        # Build a map of confirmed track_id -> detection bbox this frame
        # DeepSort attaches the matched detection via track.det_class / ltrb
        active_tracks = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id

            # to_ltrb(orig_det=True) returns the original detection bbox if
            # the track was matched this frame, otherwise the Kalman prediction.
            try:
                ltrb = track.to_ltrb(orig_det=True)
            except TypeError:
                # older versions don't support orig_det param
                ltrb = track.to_ltrb()

            active_tracks.append({
                'id': track_id,
                'bbox': [float(ltrb[0]), float(ltrb[1]), float(ltrb[2]), float(ltrb[3])]
            })
        return active_tracks
