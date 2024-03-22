import numpy as np


class Detection(object):
    """
    This class represents a bounding box detection in a single image.
    """

    def __init__(self, xyxy, confidence, class_name, feature):
        self.xyxy = np.asarray(xyxy, dtype=np.float32)
        self.confidence = float(confidence)
        self.class_name = class_name
        self.feature = np.asarray(feature, dtype=np.float32)

    def get_class(self):
        return self.class_name
    def to_xyxy(self):
        return self.xyxy.copy()
    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.xyxy.copy()
        ret[:2] += (-ret[:2] + ret[2:]) / 2
        ret[2] = (ret[2]-ret[0]) / (ret[3] - ret[1])
        ret[3] = (ret[3] - ret[1])
        return ret
