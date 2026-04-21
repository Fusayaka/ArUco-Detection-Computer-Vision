import cv2
import cv2.aruco as aruco
from ultralytics import YOLO
import numpy as np


# Pre-defined sharpening kernel
_SHARPEN_KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
 
# Gamma lookup tables pre-computed for the two fixed gamma values used
_GAMMA_TABLE = {
    gamma: np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.uint8)
    for gamma in (0.5, 2.0)
}


class HybridDetector:
    """
    Hybrid detection pipeline combining YOLOv8 (region proposal) and
    OpenCV ArUco (marker decoding) for robust single-image inference.
 
    Flow:
        1. YOLO locates candidate marker regions.
        2. Each region is cropped (with padding) and preprocessed.
        3. ArUco attempts decoding across several image variants.
        4. Detections are deduplicated and returned sorted by marker ID.
    """
 
    def __init__(self, model_path: str, conf_threshold: float = 0.5, padding: int = 20):
        """
        Args:
            model_path:      Path to trained YOLOv8 weights (.pt file).
            conf_threshold:  Minimum YOLO confidence to keep a box.
            padding:         Extra pixels added around each bounding box
                             to avoid edge-clipping the marker.
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.padding = padding
 
        self.aruco_detector = aruco.ArucoDetector(
            aruco.getPredefinedDictionary(aruco.DICT_ARUCO_MIP_36h12),
            aruco.DetectorParameters(),
        )
 
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
 
    def _get_yolo_boxes(self, image_rgb: np.ndarray) -> np.ndarray:
        """Returns YOLO bounding boxes [[x1, y1, x2, y2], ...] for the image."""
        results = self.model.predict(image_rgb, conf=self.conf_threshold, verbose=False)
        return results[0].boxes.xyxy.cpu().numpy()

    def _crop_roi(self, image_bgr: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        """
        Crops a padded ROI from *image_bgr* using the given bounding box.
 
        Returns:
            roi:    Cropped image region.
            offset: (x, y) top-left corner of the ROI in the original image,
                    used to map local coordinates back to global ones.
        """
        h, w = image_bgr.shape[:2]
        x1_f, y1_f, x2_f, y2_f = box

        x1 = max(0, int(x1_f) - self.padding)
        y1 = max(0, int(y1_f) - self.padding)
        x2 = min(w, int(x2_f) + self.padding)
        y2 = min(h, int(y2_f) + self.padding)

        offset = (float(x1), float(y1))

        return image_bgr[y1:y2, x1:x2], offset
    
    def _preprocessing_variants(self, gray: np.ndarray) -> list[np.ndarray]:
        """
        Generates a list of preprocessed images to maximise decode chances
        under varied lighting conditions.
 
        Variants (in try order):
            1. Raw grayscale
            2. CLAHE-equalised
            3. Dark-gamma correction  (gamma=0.5 — brightens)
            4. Light-gamma correction (gamma=2.0 — darkens)
            5. Sharpened CLAHE
        """
        clahe = self._clahe.apply(gray)
        return [
            gray,
            clahe,
            cv2.LUT(gray, _GAMMA_TABLE[0.5]),
            cv2.LUT(gray, _GAMMA_TABLE[2.0]),
            cv2.filter2D(clahe, -1, _SHARPEN_KERNEL),
        ]

    def _decode_aruco(self, roi: np.ndarray, offset: tuple[int, int]) -> tuple[list, np.ndarray | None]:
        """
        Tries to detect ArUco markers in *roi*, returning on the first
        successful variant to avoid unnecessary processing.
 
        Args:
            roi:    Cropped image region (BGR).
            offset: (x, y) position of the ROI in the original image.
 
        Returns:
            (corners, ids) where corners are in original-image coordinates,
            or ([], None) if no markers were found.
        """
        if roi is None or roi.size == 0:
            return [], None
 
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
 
        for variant in self._preprocessing_variants(gray):
            corners, ids, _ = self.aruco_detector.detectMarkers(variant)
            if ids is not None and len(ids) > 0:
                global_corners = [c + offset for c in corners]
                return global_corners, ids
 
        return [], None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
 
    def process_image(self, image_path: str) -> str:
        """
        Runs the full detection pipeline on a single image.
 
        Args:
            image_path: Path to the input image file.
 
        Returns:
            Space-separated string of detections sorted by marker ID:
            ``"<id> <x> <y> <id> <x> <y> ..."``
            where (x, y) is the top-left corner of each marker.
            Returns ``" "`` if the image cannot be loaded or no markers found.
        """
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return " "
 
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        boxes = self._get_yolo_boxes(img_rgb)

        markers: dict[int, tuple[float, float]] = {}
 
        for box in boxes:
            roi, offset = self._crop_roi(img_bgr, box)
            corners, ids = self._decode_aruco(roi, offset)
 
            if ids is not None:
                for marker_id, corner in zip(ids.flatten(), corners):
                    top_left = corner[0][0]
                    markers[int(marker_id)] = (top_left[0], top_left[1])
 
        if not markers:
            return " "
 
        return " ".join(
            f"{mid} {x:.3f} {y:.3f}"
            for mid, (x, y) in sorted(markers.items())
        )
 