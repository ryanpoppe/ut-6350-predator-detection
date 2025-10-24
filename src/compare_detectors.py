from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models.detection import BaseDetector
import torch
import numpy as np
from PIL import Image
import supervision as sv
import traceback


from detetctor_metrics import DetectorMetrics

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
mega_detector_5_a = pw_detection.MegaDetectorV5(device=device, pretrained=True, version="a")
mega_detector_5_b = pw_detection.MegaDetectorV5(device=device, pretrained=True, version="b")
mega_detector_6_yolov9_c = pw_detection.MegaDetectorV6(device=device, pretrained=True, version="MDV6-yolov9-c")
mega_detector_6_yolov9_e = pw_detection.MegaDetectorV6(device=device, pretrained=True, version="MDV6-yolov9-e")
mega_detector_6_yolov10_c = pw_detection.MegaDetectorV6(device=device, pretrained=True, version="MDV6-yolov10-c")
mega_detector_6_yolov10_e = pw_detection.MegaDetectorV6(device=device, pretrained=True, version="MDV6-yolov10-e")
# mega_detector_6_ultralytics_rt_detr_c = pw_detection.MegaDetectorV6_Distributed(device=device, pretrained=True, version="MDV6-rtdetr-c")
mega_detector_6_mit_yolov9_c = pw_detection.MegaDetectorV6MIT(device=device, pretrained=True, version="MDV6-mit-yolov9-c")
mega_detector_6_mit_yolov9_e = pw_detection.MegaDetectorV6MIT(device=device, pretrained=True, version="MDV6-mit-yolov9-e") 
mega_detector_6_apache_rt_detr_c = pw_detection.MegaDetectorV6Apache(device=device, pretrained=True, version="MDV6-apa-rtdetr-c")
mega_detector_6_apache_rt_detr_e = pw_detection.MegaDetectorV6Apache(device=device, pretrained=True, version="MDV6-apa-rtdetr-e")
#deepfaune_detector = pw_detection.DeepfauneDetector(device=device)
herdnet_detector = pw_detection.HerdNet(device=device, version="general")

detector_list = {
    "MegaDetectorV5-a": mega_detector_5_a,
    "MegaDetectorV5-b": mega_detector_5_b,
    "MegaDetectorV6-yolov9-c": mega_detector_6_yolov9_c,
    "MegaDetectorV6-yolov9-e": mega_detector_6_yolov9_e,
    "MegaDetectorV6-yolov10-c": mega_detector_6_yolov10_c,
    "MegaDetectorV6-yolov10-e": mega_detector_6_yolov10_e,
    # "MegaDetectorV6-Ultralytics-RtDetr-Compact": mega_detector_6_ultralytics_rt_detr_c, # produces errors
    "MegaDetectorV6-MIT-YoloV9-Compact": mega_detector_6_mit_yolov9_c,
    "MegaDetectorV6-MIT-YoloV9-Extra": mega_detector_6_mit_yolov9_e,
    "MegaDetectorV6-Apache-RTDetr-Compact": mega_detector_6_apache_rt_detr_c,
    "MegaDetectorV6-Apache-RTDetr-Extra": mega_detector_6_apache_rt_detr_e,
    #"Deepfaune": deepfaune_detector,  # Disabled because the hosting URL seems down
    "HerdNet general": herdnet_detector,
}

def main():
    images_path = "iNaturalist/Corvus_corax_images"
    manual_identifications_path = "iNaturalist/Corvus_corax_images/manual_identification.csv"

    detector_metrics = DetectorMetrics(images_path, manual_identifications_path)

    results = {}
    for detector_name, detector in detector_list.items():
        try:
            print(f"Evaluating {detector_name}...")
            metrics = detector_metrics.evaluate_detector(detector)
            results[detector_name] = metrics
            print(f"Results for {detector_name}: {metrics}\n")
        except Exception as e:
            print(f"Error evaluating {detector_name}: {type(e).__name__}: {e}\n")
            traceback.print_exc()

    print("Final comparison results:")
    for detector_name, metrics in results.items():
        """
        the metrics dict contains: 
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "accuracy": accuracy
        }
        """
        print(f"{detector_name}: Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1 Score: {metrics['f1_score']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        print(f"  False Positives ({len(metrics.get('false_positive_images', []))}): {metrics.get('false_positive_images', [])}")
        print(f"  False Negatives ({len(metrics.get('false_negative_images', []))}): {metrics.get('false_negative_images', [])}")

if __name__ == "__main__":
    main()
