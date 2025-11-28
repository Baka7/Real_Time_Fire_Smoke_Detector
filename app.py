import streamlit as st
import cv2
import tempfile
from utils import load_model, draw_boxes
from ultralytics import YOLO

st.title("📡 YOLO Live Object Detection App")

# Load model
model = load_model("best_nano_111.pt")

mode = st.radio("Choose Input Type", ["Upload Image", "Upload Video", "RTSP / Live Stream URL"])

if mode == "Upload Image":
    img = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
    if img:
        file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        results = model(frame)
        out = draw_boxes(frame, results)

        st.image(out, channels="BGR")

elif mode == "Upload Video":
    vid = st.file_uploader("Upload a video", type=["mp4","avi","mov"])

    if vid:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(vid.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            out = draw_boxes(frame, results)

            stframe.image(out, channels="BGR")

        cap.release()

elif mode == "RTSP / Live Stream URL":
    url = st.text_input("Enter RTSP / IP Camera URL or YouTube .m3u8")
    
    if url:
        cap = cv2.VideoCapture(url)
        stframe = st.empty()

        if not cap.isOpened():
            st.error("Could not open stream.")
        else:
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Stream ended or connection lost.")
                    break

                results = model(frame)
                out = draw_boxes(frame, results)

                stframe.image(out, channels="BGR")
