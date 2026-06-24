from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import List, Optional

import cv2


def _save_frames_to_mp4(frames: List[object], output_dir: Path, fps: int = 30) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    output_path = output_dir / f"browser_recording_{int(time.time())}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (width, height))
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    return output_path


def render_recorder_beta(st, output_dir: Path) -> Optional[Path]:
    st.markdown(
        '<div class="tab-intro">Beta mode: browser camera support depends on browser, device, and HTTPS/local permissions.</div>',
        unsafe_allow_html=True,
    )

    try:
        import av
        from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
    except Exception:
        st.warning(
            "Record Video Beta is unavailable because streamlit-webrtc or av is not installed. "
            "The Upload Video mode still works."
        )
        st.code("pip install streamlit-webrtc av")
        return None

    class RecordingVideoProcessor(VideoProcessorBase):
        def __init__(self) -> None:
            self.frames = []
            self.lock = threading.Lock()
            self.max_frames = 900

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="bgr24")
            with self.lock:
                self.frames.append(image.copy())
                if len(self.frames) > self.max_frames:
                    self.frames = self.frames[-self.max_frames :]
            return av.VideoFrame.from_ndarray(image, format="bgr24")

    try:
        context = webrtc_streamer(
            key="golf-shot-vision-recorder",
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=RecordingVideoProcessor,
            async_processing=True,
        )
    except Exception as exc:
        st.warning(
            "The browser camera could not be started. Use localhost, allow camera permission, "
            "or switch to Upload Video."
        )
        with st.expander("Technical details"):
            st.code(str(exc))
        return None

    if not context.video_processor:
        st.info("Start the camera stream above, then save the buffered clip when the swing is captured.")
        return None

    processor = context.video_processor
    with processor.lock:
        frame_count = len(processor.frames)
    st.caption(f"Buffered frames: {frame_count}. This Beta recorder stores the most recent frames in memory.")

    saved_path = None
    if st.button("Save buffered recording", use_container_width=True):
        with processor.lock:
            frames = list(processor.frames)
        if len(frames) < 10:
            st.warning("Not enough frames captured yet. Keep the camera running a little longer.")
        else:
            saved_path = _save_frames_to_mp4(frames, output_dir=output_dir, fps=30)
            st.session_state["recorded_video_path"] = str(saved_path)
            st.success("Recording saved. You can analyze it below.")

    if "recorded_video_path" in st.session_state:
        return Path(st.session_state["recorded_video_path"])
    return saved_path
