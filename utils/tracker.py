from deep_sort_realtime.deepsort_tracker import DeepSort

class ObjectTracker:
    def __init__(self, max_age=10, n_init=2):
        # Balanced for 60-90 FPS tracking with YOLO every 3 frames
        # n_init=2 filters out one-off false positives on whiteboards/walls
        # max_age=10 kills lost tracks quickly to prevent duplication
        self.tracker = DeepSort(
            max_age=max_age, 
            n_init=n_init, 
            max_iou_distance=0.5, 
            embedder='mobilenet',
            bgr=True
        )

    def update(self, detections, frame=None):
        """
        detections: list of ([x1, y1, w, h], confidence, label)
        """
        # Update deep_sort using internal 'mobilenet' embedder
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        active_results = []
        for track in tracks:
            # 1. Filter out unconfirmed tracks (prevents flicker/false positives)
            if not track.is_confirmed():
                continue
            
            # 2. Filter out 'ghost' tracks that haven't been seen by YOLO recently
            # With DETECTION_INTERVAL=3, a live track should have time_since_update <= 3
            if track.time_since_update > 5:
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb() # Smooth Kalman predicted box
            
            active_results.append({
                "id": track_id,
                "bbox": [float(ltrb[0]), float(ltrb[1]), float(ltrb[2]), float(ltrb[3])]
            })
            
        return active_results
