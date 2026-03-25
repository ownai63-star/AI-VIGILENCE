from deep_sort_realtime.deepsort_tracker import DeepSort

class ObjectTracker:
    def __init__(self, max_age=10, n_init=1):
        self.tracker = DeepSort(max_age=max_age, n_init=n_init, max_iou_distance=0.9)
        # Map track_id -> last known tight bbox [x1,y1,x2,y2] from actual detection
        self._det_bbox: dict = {}

    def update(self, detections, frame):
        """
        detections: list of ([x1, y1, w, h], confidence, label)
        """
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        det_ltrbs = []
        for det in detections:
            b = det[0]
            det_ltrbs.append([b[0], b[1], b[0]+b[2], b[1]+b[3]])

        active_tracks = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id

            try:
                ltrb = list(track.to_ltrb(orig_det=True))
            except TypeError:
                ltrb = list(track.to_ltrb())

            # Snapping to closest actual detection box to prevent drifting
            best_iou = 0
            best_det = ltrb
            
            for d in det_ltrbs:
                ix1 = max(ltrb[0], d[0])
                iy1 = max(ltrb[1], d[1])
                ix2 = min(ltrb[2], d[2])
                iy2 = min(ltrb[3], d[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = (ltrb[2]-ltrb[0])*(ltrb[3]-ltrb[1]) + (d[2]-d[0])*(d[3]-d[1]) - inter
                iou = inter / union if union > 0 else 0
                if iou > best_iou:
                    best_iou = iou
                    best_det = d
                    
            if best_iou > 0.1:
                ltrb = best_det
                self._det_bbox[track_id] = ltrb
            elif track_id in self._det_bbox:
                ltrb = self._det_bbox[track_id]
            else:
                self._det_bbox[track_id] = ltrb

            active_tracks.append({
                'id': track_id,
                'bbox': [float(ltrb[0]), float(ltrb[1]), float(ltrb[2]), float(ltrb[3])]
            })
        return active_tracks
