import streamlit as st
import requests
from config.config import  OUTPUT_DIR
import os

API_URL_VIDEO = "http://localhost:8000/predict_video/"
API_URL_IMAGE = "http://localhost:8000/predict_image/"

def detect_image_ui():
    st.header("📸 Detect on Image")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="image")

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_container_width =True)

        if st.button("Detect Image"):
            with st.spinner("⏳ Processing image..."):
                files = {"file": (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)}
                response = requests.post(API_URL_IMAGE, files=files)

                if response.status_code == 200:
                    st.success("✅ Image processed!")
                    st.image(response.content, caption="Detected Image", use_container_width =True)
                    result_path = os.path.join(OUTPUT_DIR, "image_result.jpg")

                    with open(result_path, "wb") as f:
                        f.write(response.content)
                    st.download_button("📥 Download Result Image", open(result_path, "rb"), file_name="result.jpg")
                else:
                    st.error("Error detecting image")

def detect_video_ui():
    st.header("🎥 Detect on Video")
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"], key="video")

    if uploaded_video is not None:
        st.video(uploaded_video)
        
        if st.button("Detect Video"):
            with st.spinner("⏳ Processing video..."):
                files = {"file": (uploaded_video.name, uploaded_video.getvalue(), uploaded_video.type)}
                response = requests.post(API_URL_VIDEO, files=files)

                if response.status_code == 200:
                    st.success("✅ Video processed successfully!")
                    st.video(response.content) 
                    result_path = os.path.join(OUTPUT_DIR, "video_result.mp4")

                    with open(result_path, "wb") as f:
                        f.write(response.content)
                    st.download_button("📥 Download Result Video", open(result_path, "rb"), file_name="result.mp4", mime="video/mp4")
                else:
                    st.error("❌ Error detecting video")

def main():
    st.set_page_config(page_title="🐛 Silkworm Disease Detection", layout="wide")

    st.markdown(
        """
        <div style="text-align:center; padding:15px; 
                    background: linear-gradient(90deg, #FFDEE9 0%, #B5FFFC 100%);
                    border-radius:12px; margin-bottom:20px;">
            <h1 style="color:#2F4F4F;">🐛 Silkworm Disease Detection</h1>
            <p style="font-size:18px;">Upload your <b>image</b> or <b>video</b> to detect segmentation masks</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.header("⚙️ Options")
    mode = st.sidebar.radio("Choose detection mode:", ["Image", "Video"], index=0)

    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👨‍💻 Author")
    st.sidebar.write("**Ho Anh Khoi**")
    st.sidebar.write("📦 Docker-ready")
    st.sidebar.write("🧪 Tested with Pytest")
    st.sidebar.write("🌐 API: FastAPI")

    if mode == "Image":
        detect_image_ui()
    else:
        detect_video_ui()



if __name__ == "__main__":
    main()