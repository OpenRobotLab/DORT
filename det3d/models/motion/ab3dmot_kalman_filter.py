# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmtrack.core import track
from mmtrack.models.motion.kalman_filter import KalmanFilter

from mmtrack.models.builder import MOTION
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints

import torch


@MOTION.register_module()
class AB3DMOTKalmanFilter(object):
    """A simple implementations of AB3DMOTKalmanFilter

    """
 

    def __init__(self):
        self.F =  np.array([[1,0,0,0,0,0,0,1,0,0],      # state transition matrix, dim_x * dim_x
		                      [0,1,0,0,0,0,0,0,1,0],
		                      [0,0,1,0,0,0,0,0,0,1],
		                      [0,0,0,1,0,0,0,0,0,0],  
		                      [0,0,0,0,1,0,0,0,0,0],
		                      [0,0,0,0,0,1,0,0,0,0],
		                      [0,0,0,0,0,0,1,0,0,0],
		                      [0,0,0,0,0,0,0,1,0,0],
		                      [0,0,0,0,0,0,0,0,1,0],
		                      [0,0,0,0,0,0,0,0,0,1]])  
        self.H = np.array([[1,0,0,0,0,0,0,0,0,0],      
		                      [0,1,0,0,0,0,0,0,0,0],
		                      [0,0,1,0,0,0,0,0,0,0],
		                      [0,0,0,1,0,0,0,0,0,0],
		                      [0,0,0,0,1,0,0,0,0,0],
		                      [0,0,0,0,0,1,0,0,0,0],
		                      [0,0,0,0,0,0,1,0,0,0]])

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Args:
            measurement (ndarray):  Bounding box coordinates (x, y, a, h) with
            center position (x, y), aspect ratio a, and height h.

        Returns:
             (ndarray, ndarray): Returns the mean vector (8 dimensional) and
                covariance matrix (8x8 dimensional) of the new track.
                Unobserved velocities are initialized to 0 mean.
        """

        kf = KalmanFilter(dim_x=10, dim_z=7)

        kf.F = self.F
        kf.H = self.H
		# initial state uncertainty at time 0
		# Given a single data, the initial velocity is very uncertain, so giv a high uncertainty to start
        kf.P[7:, 7:] *= 1000.
        kf.P *= 10.
        kf.Q[7:, 7:] *= 0.01
        if isinstance(measurement, list):
            measurement = measurement[0]
        kf.x[:7] = measurement.T[:7].cpu().numpy()
        return kf

    def predict(self, kf, ):
        """Run Kalman filter prediction step.

        Args:
            mean (ndarray): The 8 dimensional mean vector of the object
                state at the previous time step.

            covariance (ndarray): The 8x8 dimensional covariance matrix
                of the object state at the previous time step.

        Returns:
            (ndarray, ndarray): Returns the mean vector and covariance
                matrix of the predicted state. Unobserved velocities are
                initialized to 0 mean.
        """
        kf.predict()
        return kf

    def update(self, kf, measurement):
        """Run Kalman filter correction step.

        Args:
            mean (ndarray): The predicted state's mean vector (8 dimensional).
            covariance (ndarray): The state's covariance matrix (8x8
                dimensional).
            measurement (ndarray): The 4 dimensional measurement vector
                (x, y, a, h), where (x, y) is the center position, a the
                aspect ratio, and h the height of the bounding box.


        Returns:
             (ndarray, ndarray): Returns the measurement-corrected state
             distribution.
        """
        if isinstance(measurement, torch.Tensor):
            measurement = measurement.T.cpu().numpy()
        else:
            raise NotImplementedError
        kf.update(measurement[:7])

        return kf

    def track(self, tracks, bboxes):
        """Track forward.

        Args:
            tracks (dict[int:dict]): Track buffer.
            bboxes (Tensor): Detected bounding boxes.

        Returns:
            (dict[int:dict], Tensor): Updated tracks and bboxes.
        """
        for id, track in tracks.items():
            track.kf = self.predict(track.kf)

        # tracks.kf = self.predict(tracks.kf)
        # tracks.bboxes = track.bboxes
        costs = None
        return tracks, costs

