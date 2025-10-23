import pandas as pd
import os

from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models.detection import BaseDetector




class DetectorMetrics:
    def __init__(self, images_path: str, manual_identifications_path: str):
        self._images_path = images_path
        self._manual_identifications_path = manual_identifications_path
        self._images_list = self.get_images_list()
        self._manual_identifications = self.load_manual_identifications()
        self._detector_class_names = {
            0: "animal",
            1: "person",
            2: "vehicle"
        }

    def get_images_list(self) -> list[str]:
        images = []
        for file_name in os.listdir(self._images_path):
            if file_name.lower().endswith(('.jpeg')):
                images.append(os.path.join(self._images_path, file_name))
        return images
    
    def load_manual_identifications(self) -> pd.DataFrame:
        df = pd.read_csv(self._manual_identifications_path)
        return df
    
    def evaluate_detector(self, detector: BaseDetector) -> dict:
        # Run the detector on the images and compare with manual identifications
        # get all the image filenames from the manual identifications
        image_filenames = self._manual_identifications['image_filename'].tolist()
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0
        false_positive_images = set()
        false_negative_images = set()

        for image_filename in image_filenames:
            image_path = os.path.join(self._images_path, image_filename)
            if not os.path.exists(image_path):
                continue
            
            results = detector.single_image_detection(image_path)

            manual_entry = self._manual_identifications[self._manual_identifications['image_filename'] == image_filename]
            if manual_entry.empty:
                continue
            manual_identification = manual_entry['identification'].values[0]
            # the manual identification can be "animal", "none", or "Dead Tortoise Skeleton"
            for i, (xyxy, det_id) in enumerate(zip(results["detections"].xyxy, results["detections"].class_id)):
                detected_class = self._detector_class_names.get(det_id, "unknown")
                if detected_class == "animal":
                    if manual_identification == "animal":
                        true_positives += 1
                    else:
                        false_positives += 1
                        false_positive_images.add(image_filename)
                else:
                    if manual_identification == "animal":
                        false_negatives += 1
                        false_negative_images.add(image_filename)
                    else:
                        true_negatives += 1

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives) if (true_positives + true_negatives + false_positives + false_negatives) > 0 else 0
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "accuracy": accuracy,
            "false_positive_images": sorted(false_positive_images),
            "false_negative_images": sorted(false_negative_images)
        }
