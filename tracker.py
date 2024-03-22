from __future__ import absolute_import
import numpy as np
from kalman_filter import KalmanFilter
import linear_sum_assignment
import iou_matching
from track import Track

class Tracker:
    """
    This is the multi-target tracker
    __________
    args:
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.
    """

    def __init__(self, metric, max_iou_distance = 0.7, max_age = 30, n_init =3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.kf = KalmanFilter()
        self.tracks = []
        self.next_id = 1
    def predict(self):
        """Propagate track state distribution one time step foward
        so should be called once every time step, before 'update'"""
        for track in self.tracks:
            track.predict(self.kf)


    def update(self, detections):
        matches, unmatched_tracks, unmatched_detections = self.match(detections)

        #update tracks set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[detection_idx]) # update feature, state
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        for detection_idx in unmatched_detections:
            self.initiate_track(detections[detection_idx]) #initate new track 

        self.tracks = [t for t in self.tracks if not t.is_deleted()] # giữ lại các track chưa bị xóa

        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()] #lấy ra các track_id đã được confirmed
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features  # lấy feature and ID of track được xác nhận
            targets += [track.track_id for i in track.features]
            track.features = []
        #update distance metric
        self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets)
                                        # features               track_id , track_id confirmed
    def match(self,detections):
        def gated_metric(tracks, dets, track_indices, detection_indices):
            """compute cost có điều kiện between the features of the detections and the corresponding track IDs."""
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_sum_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices, detection_indices)
            # delete các detections have cost > predict of kalman_filter về vị trí của track
            return cost_matrix
        
        confirmed_tracks = [ i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i,t in enumerate(self.tracks) if not t.is_confirmed()]
        # Performs cascade matching for confirmed tracks.
        matches_a , unmatched_tracks_a, unmatched_detections = linear_sum_assignment.matching_cascade(gated_metric, self.metric.matching_threshold, self.max_age, self.tracks, detections, confirmed_tracks)
        # return match_a: các confirmed tracks được ghép thành công với detection
        # unmatched_tracks_a , unmatched_detections

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [ k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [ k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = linear_sum_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)
        
        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    # chúng ta thực hiện 2 bước tìm match:
        # sử dụng matching_cascade với gate_metric compute cost base on feature của detection và track_id . Nó phù hợp với các confirm trác
        # sử dụng min_cost_matching với iou_cost giữa bounding box của các remaining track và detection.
            #min_cost_matching tìm kiếm các ghép nối có cost IoU thấp nhất, đảm bảo sử dụng hiệu quả các detection còn lại.
    
    def initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyxy())
        class_name = detection.get_class()
        self.tracks.append(Track(
            mean, covariance, self.next_id, self.n_init, self.max_age,detection.feature, class_name))
        self.next_id += 1

