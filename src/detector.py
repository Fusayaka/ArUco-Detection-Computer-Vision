import cv2
import cv2.aruco as aruco
from ultralytics import YOLO

class HybridDetector:
    """
    A hybrid detection pipeline integrating deep learning (YOLOv8) 
    and traditional computer vision (OpenCV) for robust ArUco marker detection.
    """
    def __init__(self, model_path, conf_threshold=0.5, padding=20):
        """
        Initializes the hybrid detector.
        
        Args:
            model_path (str): Path to the trained YOLOv8 weights (.pt file).
            conf_threshold (float): Confidence threshold for YOLO predictions.
            padding (int): Pixel margin added to bounding boxes to prevent edge cropping.
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.padding = padding
        self.detector = aruco.ArucoDetector(
            aruco.getPredefinedDictionary(aruco.DICT_ARUCO_MIP_36h12),
            aruco.DetectorParameters()
        )

    def _get_yolo_regions(self, image):
        """
        Predicts bounding boxes for candidate ArUco markers using the YOLO model.
        
        Args:
            image (numpy.ndarray): The input image in RGB format.
            
        Returns:
            numpy.ndarray: Array of bounding box coordinates [x1, y1, x2, y2].
        """
        results = self.model.predict(image, conf=self.conf_threshold, verbose=False)
        return results[0].boxes.xyxy.cpu().numpy()

    def _extract_roi(self, image, box):
        """
        Extracts the Region of Interest (ROI) from the original image with padding.
        
        Args:
            image (numpy.ndarray): The original input image.
            box (list or numpy.ndarray): Bounding box coordinates [x1, y1, x2, y2].
            
        Returns:
            tuple: (Cropped ROI image, Offset coordinates (x, y) for global mapping).
        """
        height, width, _ = image.shape
        x1, y1, x2, y2 = box

        x1_p = max(0, int(x1 - self.padding))
        y1_p = max(0, int(y1 - self.padding))
        x2_p = min(width, int(x2 + self.padding))
        y2_p = min(height, int(y2 + self.padding))

        roi = image[y1_p:y2_p, x1_p:x2_p]
        return roi, (x1_p, y1_p)

    def _decode_aruco(self, roi, offset):
        """
        Detects ArUco markers within the ROI and remaps local coordinates to global scale.
        
        Args:
            roi (numpy.ndarray): The cropped image region containing the marker.
            offset (tuple): The (x, y) coordinates of the ROI's top-left corner in the original image.
            
        Returns:
            tuple: (List of global corner coordinates, Array of detected marker IDs).
        """
        corners, ids, _ = self.detector.detectMarkers(roi)
        real_corners = []
        if ids is not None:
            for c in corners:
                real_corners.append(c + offset)
        return real_corners, ids

    def process_image(self, image_path):
        """Executes the complete detection pipeline and returns the formatted prediction string."""
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return " "
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        boxes = self._get_yolo_regions(img_rgb)
        
        all_markers = []
        
        for box in boxes:
            roi, offset = self._extract_roi(img_bgr, box)
            corners, ids = self._decode_aruco(roi, offset)
            
            if ids is not None:
                for i in range(len(ids)):
                    marker_id = int(ids[i][0])
                    top_left = corners[i][0][0]
                    all_markers.append((marker_id, top_left[0], top_left[1]))
                    
        all_markers.sort(key=lambda x: x[0])
        
        output = [f"{m[0]} {m[1]:.3f} {m[2]:.3f}" for m in all_markers]
        return " ".join(output) if output else " "