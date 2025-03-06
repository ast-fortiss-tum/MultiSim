from opensbt.quality_indicators.utils.distance_indicator_sets import DistanceIndicator, euclidean_distance


class IGDE(DistanceIndicator):

    def __init__(self, Z, **kwargs):
        super().__init__(Z, euclidean_distance, 1, **kwargs)
