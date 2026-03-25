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
            
            # If the track wasn't matched to a real YOLO detection THIS frame, drop it instantly
            # to prevent ghost boxes floating in empty space.
            if getattr(track, 'time_since_update', 0) > 0:
                continue

            track_id = track.track_id

            try:
                ltrb = list(track.to_ltrb(orig_det=True))
            except TypeError:
                ltrb = list(track.to_ltrb())
                
            ltrb_flt = [float(x) for x in ltrb]

            # Snapping to closest actual detection box to prevent drifting
            best_iou = 0
            best_det = ltrb
            
            for d in det_ltrbs:
                df = [float(x) for x in d]
                ix1 = max(ltrb_flt[0], df[0])
                iy1 = max(ltrb_flt[1], df[1])
                ix2 = min(ltrb_flt[2], df[2])
                iy2 = min(ltrb_flt[3], df[3])
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                union = (ltrb_flt[2]-ltrb_flt[0])*(ltrb_flt[3]-ltrb_flt[1]) + (df[2]-df[0])*(df[3]-df[1]) - inter
                union_flt = float(union)
                iou = float(inter) / union_flt if union_flt > 0.0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_det = df
                    
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
