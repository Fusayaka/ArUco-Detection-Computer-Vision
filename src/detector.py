import cv2
import cv2.aruco as aruco
from ultralytics import YOLO
import numpy as np
import os

from src.corners import *
from src.decode import *
from src.preprocess import enhance_image

# Pre-defined sharpening kernel
_SHARPEN_KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
 
# Gamma lookup tables pre-computed for the two fixed gamma values used
_GAMMA_TABLE = {
    gamma: np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.uint8)
    for gamma in (0.5, 2.0)
}

@dataclass
class PipelineResult:
    """One fully decoded marker detected in an image."""

    marker_id: int  # Matched ID in ARUCO_MIP_36H12 (0–249)
    top_left_x: float  # Canonical top-left corner, x pixel coordinate
    top_left_y: float  # Canonical top-left corner, y pixel coordinate
    hamming: int  # Hamming distance of the decode (0 = perfect)
    corners: np.ndarray  # Refined (4,2) corners in image coords

def _canonical_top_left(
    refined_corners: np.ndarray,
    rotation: int,
) -> tuple[float, float]:
    """Return the canonical top-left corner (x, y) in image pixel coordinates.

    The refined_corners array is in spatial order: [TL, TR, BR, BL] by image
    position.  The decode rotation maps which spatial corner is the canonical
    (marker-readable) top-left.  See module docstring for the full derivation.

    Args:
        refined_corners: (4, 2) float32 — output of refine_corners().
        rotation:        decode_result.rotation — 0/1/2/3.

    Returns:
        (x, y) of the canonical top-left corner.
    """
    tl = refined_corners[rotation % 4]
    return float(tl[0]), float(tl[1])

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

        self.corner_model = load_corner_model(checkpoint_path="models/best_corners.pth")
 
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

    def expand_bbox(
        self,
        bbox: tuple[int, int, int, int],
        img_h: int,
        img_w: int,
        margin: float = 0.20,
    ) -> tuple[int, int, int, int]:
        """Expand a bounding box by a relative margin on all sides, clamped to
        the image boundaries.

        WHY 20% MARGIN?
        ---------------
        The corner-refinement CNN needs to *see the corners*, not just the
        interior of the marker.  If we crop exactly to the detected bounding box,
        any positional error in the detection will clip off one or more corners.
        The DeepArUco++ paper uses a 20% margin and reports that it reliably
        keeps all four corners in the crop even under moderate detection error.

        Args:
            bbox:   (x_min, y_min, x_max, y_max) tight bounding box.
            img_h:  Image height in pixels (clamp upper bound).
            img_w:  Image width in pixels (clamp upper bound).
            margin: Fractional expansion per side (0.20 = 20%).

        Returns:
            Expanded (x_min, y_min, x_max, y_max), clamped to image bounds.
        """
        x0, y0, x1, y1 = bbox
        dx = int((x1 - x0) * margin)
        dy = int((y1 - y0) * margin)
        return (
            max(0, x0 - dx),
            max(0, y0 - dy),
            min(img_w, x1 + dx),
            min(img_h, y1 + dy),
        )


    def crop_detection(
        self,
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
        target_size: int = 64,
        margin: float = 0.20,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Crop a (possibly expanded) bounding-box region and resize to target_size.

        WHY 64×64?
        ----------
        The corner-refinement CNN (Section 4.2 of DeepArUco++) expects exactly
        64×64 input.  Smaller inputs lose detail; larger inputs waste memory and
        slow inference.  64px is large enough to resolve individual cells of the
        6×6 marker grid even for small markers (~30px wide at native resolution).

        Args:
            image:       Full BGR or grayscale image.
            bbox:        Tight (x_min, y_min, x_max, y_max).
            target_size: Side length to resize the crop to.
            margin:      Fractional margin added before cropping.

        Returns:
            (crop, expanded_bbox) where crop is uint8 shape
            (target_size, target_size, C) and expanded_bbox is the actual pixel
            region that was cropped (needed to map corners back to image coords).
        """
        h, w = image.shape[:2]
        exp_bbox = self.expand_bbox(bbox, h, w, margin)
        x0, y0, x1, y1 = exp_bbox

        crop = image[int(y0):int(y1), int(x0):int(x1)]
        if crop.size == 0:
            # Degenerate crop (bbox outside image) — return blank
            if image.ndim == 3:
                crop = np.zeros((target_size, target_size, image.shape[2]), dtype=np.uint8)
            else:
                crop = np.zeros((target_size, target_size), dtype=np.uint8)
            return crop, exp_bbox

        crop = cv2.resize(crop, (target_size, target_size),
                        interpolation=cv2.INTER_LINEAR)
        return crop, exp_bbox

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

        # offset = (float(x1), float(y1))
        offset = (float(x1), float(y1), float(x2), float(y2))

        return image_bgr[y1:y2, x1:x2], offset
        # return image_bgr[y1:y2, x1:x2], (x1, y1, x2, y2)
    
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

    def _decode_aruco(self, roi: np.ndarray, offset: tuple[int, int, int, int]) -> tuple[list, np.ndarray | None]:
        """
        Tries to detect ArUco markers in *roi*, returning on the first
        successful variant to avoid unnecessary processing.
 
        Args:
            roi:    Cropped image region (BGR).
            offset: (x, y, x_max, y_max) position of the ROI in the original image.
 
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
                global_corners = [c + offset[:2] for c in corners]
                return global_corners, ids
 
        return [], None

    def _format_prediction_string(self, results: list[PipelineResult]) -> str:
        """Convert a list of PipelineResults to the Kaggle submission string.

        Format: "id x y id x y ..."
        Example: "29 481.785 261.833 102 273.434 321.559"

        If results is empty the prediction string is an empty string, which
        tells the scorer there are no detections in this image.  That is the
        correct behaviour for truly empty images (giving score = 1 if the
        ground truth also has no markers, per the assignment spec).

        WHY ROUND TO 3 DECIMAL PLACES?
        The scorer computes Euclidean distance in pixels.  Three decimal places
        gives sub-pixel precision (0.001 px), which is more than sufficient
        given that the CNN itself has ~1 px accuracy.  More decimal places waste
        bandwidth without improving the score.
        """
        if not results:
            return " "

        parts = []
        for r in results:
            parts.append(f"{r.marker_id} {r.top_left_x:.3f} {r.top_left_y:.3f}")

        return " ".join(parts)
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

        img_bgr = enhance_image(img_bgr, correct_gradient=True)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        boxes = self._get_yolo_boxes(img_rgb)

        markers: dict[int, tuple[float, float]] = {}
 
        results = []
        for box in boxes:
            crop, exp_bbox = self.crop_detection(img_bgr, box)
            
            # roi, offset = self._crop_roi(image_bgr=img_bgr, box=box)
            
            refined_corners = refine_corners_cnn(crop, exp_bbox, self.corner_model)

            decode_result = decode_marker(img_bgr, refined_corners)
        #     results.append((exp_bbox, refined_corners, offset))

        # return " ".join(f"exp_bbox: {exp_bbox[0]:.3f} {exp_bbox[1]:.3f} {exp_bbox[2]:.3f} {exp_bbox[3]:.3f} \n refined corners: {refined_corners} \n roi_offset: {offset[0]:.3f} {offset[1]:.3f} {offset[2]:.3f} {offset[3]:.3f} \n -------------------------------------------------- \n" for exp_bbox, refined_corners, offset in results)

            # cropped_corners = img_bgr[int(refined_corners[:, 1].min()):int(refined_corners[:, 1].max()), int(refined_corners[:, 0].min()):int(refined_corners[:, 0].max())]
            # # corners, ids = self._decode_aruco(corners_cropped, [0, 0])
            # warped = warp_marker(img_bgr, refined_corners)
            # normalized = normalize_patch(warped)
            # decoded_corners, ids = self._decode_aruco(warped, (refined_corners[0][0], refined_corners[0][1]))
            # results.append(decode_result)

        # return " ".join(f"decoded_corners: {decoded_corners} ids: {ids}" for decoded_corners, ids in results)
            if decode_result is not None:
                tl_x, tl_y = _canonical_top_left(refined_corners, decode_result.rotation)

                results.append(
                    PipelineResult(
                        marker_id=decode_result.marker_id,
                        top_left_x=tl_x,
                        top_left_y=tl_y,
                        hamming=decode_result.hamming,
                        corners=refined_corners,
                    )
                )

        return self._format_prediction_string(results)

        #     def ensure_bgr(img: np.ndarray) -> np.ndarray:
        #         if img.ndim == 2:
        #             return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #         return img

        #     img1 = ensure_bgr(crop)
        #     img2 = ensure_bgr(cropped_corners)
        #     img3 = ensure_bgr(warped)

        #     target_h = 300
        #     def resize_to_height(img: np.ndarray, h: int) -> np.ndarray:
        #         scale = h / img.shape[0]
        #         w = max(1, int(img.shape[1] * scale))
        #         return cv2.resize(img, (w, h))

        #     panel = np.hstack([
        #         resize_to_height(ensure_bgr(crop),            target_h),
        #         resize_to_height(ensure_bgr(cropped_corners), target_h),
        #         resize_to_height(ensure_bgr(warped),          target_h),
        #     ])
        #     # Label each column
        #     for i, label in enumerate(["crop", "corners_cropped", "warped"]):
        #         x = i * (panel.shape[1] // 3) + 5
        #         cv2.putText(panel, label, (x, 20),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        #     cv2.imshow("Comparison", panel)
        #     cv2.waitKey(0)

        # cv2.destroyAllWindows()
            
        # if not results:
        #     return " "

        # for r in results:
        #     filename = os.path.splitext(os.path.basename(image_path))[0]
        #     cv2.imshow(filename, r)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        # return " ".join(f"{r}" for r in results)

        # parts = []

        # for r in results:
        #     parts.append(f"{r.marker_id} {r.top_left_x:.3f} {r.top_left_y:.3f}")

        # return " ".join(parts)

        # ===================================================================
        # YOLOv8 + ArUco
        # ===================================================================
            # roi, offset = self._crop_roi(image_bgr=img_bgr, box=box)
            # corners, ids = self._decode_aruco(roi, offset)
 
        #     if ids is not None:
        #         for marker_id, corner in zip(ids.flatten(), corners):
        #             top_left = corner[0][0]
        #             markers[int(marker_id)] = (top_left[0], top_left[1])
 
        # if not markers:
        #     return " "
 
        # return " ".join(
        #     f"{mid} {x:.3f} {y:.3f}"
        #     for mid, (x, y) in sorted(markers.items())
        # )
 