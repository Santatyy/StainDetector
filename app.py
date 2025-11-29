import streamlit as st
import cv2
import numpy as np

class CleanlinessDetector:
    def __init__(self, image_file):
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        self.img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if self.img is None:
            raise ValueError("Error loading image")

    def process(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        binary_adaptive = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        kernel = np.ones((3,3), np.uint8)
        cleaned_mask = cv2.morphologyEx(binary_adaptive, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return cleaned_mask

    def calculate_score(self, mask):
        dirty_pixels = cv2.countNonZero(mask)
        total_pixels = mask.shape[0] * mask.shape[1]
        
        dirty_percentage = (dirty_pixels / total_pixels) * 100
        clean_percentage = 100 - dirty_percentage
        
        return clean_percentage, dirty_percentage

st.set_page_config(page_title="Surface Stain Detector", layout="centered")

st.title("Surface Stain detector")
st.caption("© 2025 Pannawit Sripratchayaprapha")
st.write("Upload an image to detect contamination using **Computer Vision (OpenCV)**.")
st.info("Real-world application: Quality control in manufacturing, ensuring surfaces are free of defects or contaminants before processing.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    detector = CleanlinessDetector(uploaded_file)
    
    st.image(uploaded_file, caption="Original Image", use_container_width=True)
    st.write("Processing...")
    
    mask = detector.process()
    clean_pct, dirty_pct = detector.calculate_score(mask)

    st.image(mask, caption="Stain detected", use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("Cleanliness Score", f"{clean_pct:.2f}%")
    col2.metric("Contamination", f"{dirty_pct:.2f}%", delta_color="inverse")

    if clean_pct > 85:
        st.success("🙂")
    else:
        st.error("🙏 😭")