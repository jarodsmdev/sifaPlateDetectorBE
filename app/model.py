from open_image_models import LicensePlateDetector

detector = LicensePlateDetector(
    detection_model="yolo-v9-t-256-license-plate-end2end"
)

def detect_plate(image_path: str):
    return detector.predict(image_path)