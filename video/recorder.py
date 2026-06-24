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
        '<div class="tab-intro">Beta 功能：浏览器摄像头支持取决于设备、浏览器，以及 HTTPS 或本地权限环境。</div>',
        unsafe_allow_html=True,
    )

    try:
        import av
        from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
    except Exception:
        st.warning(
            "实时录制 Beta 暂不可用：未安装 streamlit-webrtc 或 av。"
            "上传视频模式仍然可以正常使用。"
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
            "无法启动浏览器摄像头。请确认已允许摄像头权限，使用 HTTPS 或 localhost，"
            "也可以切换到上传视频模式。"
        )
        with st.expander("技术详情"):
            st.code(str(exc))
        return None

    if not context.video_processor:
        st.info("请先启动上方摄像头画面，挥杆录制完成后再保存缓存视频。")
        return None

    processor = context.video_processor
    with processor.lock:
        frame_count = len(processor.frames)
    st.caption(f"已缓存帧数：{frame_count}。Beta 录制会在内存中保存最近的视频帧。")

    saved_path = None
    if st.button("保存缓存录制", use_container_width=True):
        with processor.lock:
            frames = list(processor.frames)
        if len(frames) < 10:
            st.warning("已捕获帧数不足，请让摄像头继续运行一会儿。")
        else:
            saved_path = _save_frames_to_mp4(frames, output_dir=output_dir, fps=30)
            st.session_state["recorded_video_path"] = str(saved_path)
            st.success("录制已保存，可以在下方开始分析。")

    if "recorded_video_path" in st.session_state:
        return Path(st.session_state["recorded_video_path"])
    return saved_path
