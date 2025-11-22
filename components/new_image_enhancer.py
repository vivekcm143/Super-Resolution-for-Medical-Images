import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

from model.run_inference import run_model_inference, prepare_image
from model.model_config import DEVICE

def new_image_enhancer_UI(model):
    st.divider()
    st.subheader("Choose a Low Resolution X-Ray image ...")
    st.caption("⚠ Check Solution Risks in the Project Home Page")

    image_upload = st.file_uploader("Upload A Low Resolution X-Ray Image", type=["png", "jpg", "jpeg", "bmp", "tiff"])

    if image_upload is not None:
        try:
            input_image = Image.open(image_upload).convert('RGB')
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return

        input_image_cv = np.array(input_image)
        input_image_cv = cv2.cvtColor(input_image_cv, cv2.COLOR_RGB2BGR)

        ## Bilateral filter parameters
        d = st.slider("Diameter of pixel neighborhood:", 5, 15, 9)
        sigma_color = st.slider("Sigma Color:", 50, 150, 75)
        sigma_space = st.slider("Sigma Space:", 50, 150, 75)

        ## Apply bilateral filtering
        bilateral_filtered_cv = cv2.bilateralFilter(input_image_cv, d, sigma_color, sigma_space)
        bilateral_filtered = Image.fromarray(cv2.cvtColor(bilateral_filtered_cv, cv2.COLOR_BGR2RGB))

        ## Apply Gaussian filtering
        gaussian_filtered_cv = cv2.GaussianBlur(input_image_cv, (5, 5), 0)
        gaussian_filtered = Image.fromarray(cv2.cvtColor(gaussian_filtered_cv, cv2.COLOR_BGR2RGB))

        ## Apply Median filtering
        median_filtered_cv = cv2.medianBlur(input_image_cv, 5)
        median_filtered = Image.fromarray(cv2.cvtColor(median_filtered_cv, cv2.COLOR_BGR2RGB))

        ## Prepare the input images for the model
        def prepare_buffer(image):
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)
            return buffer

        image_buffer_raw = prepare_buffer(input_image)
        image_buffer_bilateral = prepare_buffer(bilateral_filtered)
        image_buffer_gaussian = prepare_buffer(gaussian_filtered)
        image_buffer_median = prepare_buffer(median_filtered)

        device = DEVICE if DEVICE else "cpu"

        ## Run super-resolution model
        raw_super_image = run_model_inference(prepare_image(image_buffer_raw, is_hr_image=True), model, device=device)
        bilateral_super_image = run_model_inference(prepare_image(image_buffer_bilateral, is_hr_image=True), model, device=device)
        gaussian_super_image = run_model_inference(prepare_image(image_buffer_gaussian, is_hr_image=True), model, device=device)
        median_super_image = run_model_inference(prepare_image(image_buffer_median, is_hr_image=True), model, device=device)

        ## Calculate PSNR and SSIM
        def calculate_metrics(original, processed):
            processed_resized = cv2.resize(processed, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_CUBIC)
            data_range = max(1e-5, processed_resized.max() - processed_resized.min())
            return psnr(original, processed_resized, data_range=data_range), ssim(original, processed_resized, channel_axis=-1, data_range=data_range)

        psnr_raw, ssim_raw = calculate_metrics(input_image_cv.astype(np.float32), np.array(raw_super_image).astype(np.float32))
        psnr_bilateral, ssim_bilateral = calculate_metrics(bilateral_filtered_cv.astype(np.float32), np.array(bilateral_super_image).astype(np.float32))
        psnr_gaussian, ssim_gaussian = calculate_metrics(gaussian_filtered_cv.astype(np.float32), np.array(gaussian_super_image).astype(np.float32))
        psnr_median, ssim_median = calculate_metrics(median_filtered_cv.astype(np.float32), np.array(median_super_image).astype(np.float32))

        ## Display input and first two outputs in the same row
        cols_input_top = st.columns(3)
        with cols_input_top[0]:
            st.subheader("🟡 **Input Image**")
            st.image(input_image, use_column_width=True)
        with cols_input_top[1]:
            st.subheader("🔴 **Unprocessed SR Image**")
            st.image(raw_super_image, use_column_width=True)
            st.markdown(f"📌 **PSNR:** `{psnr_raw:.2f} dB`  \n📌 **SSIM:** `{ssim_raw:.4f}`")
        with cols_input_top[2]:
            st.subheader("🔵 **Bilateral Filtered SR**")
            st.image(bilateral_super_image, use_column_width=True)
            st.markdown(f"📌 **PSNR:** `{psnr_bilateral:.2f} dB`  \n📌 **SSIM:** `{ssim_bilateral:.4f}`")

        ## Display the other two images in the next row
        cols_bottom = st.columns(2)
        with cols_bottom[0]:
            st.subheader("🟢 **Gaussian Filtered SR**")
            st.image(gaussian_super_image, use_column_width=True)
            st.markdown(f"📌 **PSNR:** `{psnr_gaussian:.2f} dB`  \n📌 **SSIM:** `{ssim_gaussian:.4f}`")
        with cols_bottom[1]:
            st.subheader("🟠 **Median Filtered SR**")
            st.image(median_super_image, use_column_width=True)
            st.markdown(f"📌 **PSNR:** `{psnr_median:.2f} dB`  \n📌 **SSIM:** `{ssim_median:.4f}`")