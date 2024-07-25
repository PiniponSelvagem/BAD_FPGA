import os
from collections import Counter
from pathlib import Path

from detection import RNNDetector
from domain.track import Track
from utils import misc_utils

class RNNDetectorValidator:

    def __init__(self, detector:RNNDetector):
        self.detector = detector

    def computeMetricsAgainstAnnotatedDirectory(self, directory_path :str) -> Counter:
        print(f'Computing metrics for files in directory: {directory_path}')
        metrics_dir = Counter()
        entries = Path(directory_path)
        for entry in entries.iterdir():
            metrics_dir += self.computeMetricsAgainstAnnotatedDirectory(os.path.join(directory_path, entry.name)) \
                           if entry.is_dir() else self.computeMetricsAgainstAnnotatedFile(f'{directory_path}/{entry.name}')
        return metrics_dir

    def computeMetricsAgainstAnnotatedFile(self, json_file_path :str) -> Counter:
        media_file_annotation = self._load_json_annotation_file(json_file_path)
        return self._compute_metrics_of_prediction_against_annotation(media_file_annotation)

    def _load_json_annotation_file(self, json_file_path :str) -> dict :
        return misc_utils.read_json_file(json_file_path)

    def _compute_metrics_of_prediction_against_annotation(self, media_file_annotation:dict) -> Counter:
        track = Track()
        metrics_counter = []
        metrics_counter += map(lambda track_elmt : track.compute_metrics_of_prediction_against_annotation(track_elmt) ,
                               media_file_annotation.get("tracks"))
        return misc_utils.convert_counter_collection_to_counter(metrics_counter)



# if __name__ == '__main__' :
#     detector = RNNDetector()
#     validator = RNNDetectorValidator(detector)
#     # metrics = validator.computeMetricsAgainstAnnotatedFile( os.path.abspath(os.path.join(os.path.dirname(__file__), "media-annotation/SWIFT_20000101_022052.json")) )
#     metrics = validator.computeMetricsAgainstAnnotatedDirectory( os.path.abspath(os.path.join(os.path.dirname(__file__), "media-annotation")) )
#     print(f'Accuracy  : {misc_utils.getAccuracy(metrics)}')
#     print(f'Precision : {misc_utils.getPrecision(metrics)}')
#     print(f'Recall    : {misc_utils.getRecall(metrics)}')
#     print(f'F1        : {misc_utils.getF1(metrics)}')
#     print(f'Total METRICS : {metrics}')