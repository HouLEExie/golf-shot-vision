from __future__ import annotations

import json
import math
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

import streamlit as st

from detection.ball_detector import BallDetector, Detection
from detection.motion_filter import select_best_trajectory
from physics.carry_estimator import estimate_carry
from physics.launch_angle import estimate_launch_angle
from physics.speed_estimator import estimate_initial_ball_speed
from styles import get_css
from tracking.trajectory_cleaner import (
    TrajectoryPoint,
    compute_confidence,
    smooth_trajectory,
)
from video.loader import (
    VideoLoadError,
    ensure_output_dirs,
    get_video_metadata,
    iter_video_frames,
    save_uploaded_file,
)
from video.recorder import render_recorder_beta
from visualization.overlay import render_traced_video
from visualization.plots import save_trajectory_plot


APP_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = APP_ROOT / "outputs"
UPLOAD_DIR = OUTPUT_ROOT / "uploads"
TRACE_DIR = OUTPUT_ROOT / "traces"
REPORT_DIR = OUTPUT_ROOT / "reports"
PLOT_DIR = OUTPUT_ROOT / "plots"
RECORDING_DIR = OUTPUT_ROOT / "recordings"


TRAIL_COLORS = {
    "Electric Blue": "#00D5FF",
    "Neon Green": "#39FF88",
    "Golden Yellow": "#FFD23F",
    "Hot Red": "#FF3B30",
    "White Glow": "#F8FBFF",
}


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    if len(value) != 6:
        return (255, 255, 255)
    red = int(value[0:2], 16)
    green = int(value[2:4], 16)
    blue = int(value[4:6], 16)
    return (blue, green, red)


def metric_card(label: str, value: str, detail: str) -> str:
    return f"""
    <div class="metric-card">
        <span>{label}</span>
        <strong>{value}</strong>
        <small>{detail}</small>
    </div>
    """


def render_hero() -> None:
    st.markdown(
        """
        <section class="hero-panel">
            <div class="hero-kicker">Golf Shot Vision</div>
            <h1>Golf Shot Vision</h1>
            <h2>iPhone slow-motion golf shot tracer</h2>
            <p>Upload or record your golf swing video, trace the ball flight, and estimate carry distance.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def sidebar_settings() -> tuple[Dict[str, object], Dict[str, object]]:
    with st.sidebar:
        st.markdown("## Analysis Settings")
        fps = st.number_input("FPS", min_value=24, max_value=1000, value=240, step=1)
        phone_distance_m = st.number_input(
            "Phone to ball distance (m)",
            min_value=0.5,
            max_value=30.0,
            value=4.0,
            step=0.5,
        )
        reference_length_m = st.number_input(
            "Reference object real length (m)",
            min_value=0.05,
            max_value=20.0,
            value=1.0,
            step=0.05,
        )
        reference_pixels = st.number_input(
            "Reference object pixel length (px)",
            min_value=1,
            max_value=10000,
            value=300,
            step=10,
        )
        club_type = st.selectbox(
            "Club type",
            ["Driver", "Wood", "Hybrid", "Iron", "Wedge"],
            index=0,
        )
        camera_angle = st.selectbox(
            "Camera angle",
            ["Side View", "Rear View"],
            index=0,
        )

        st.markdown("---")
        st.markdown("## Trail Settings")
        color_choice = st.selectbox(
            "Trail color",
            [
                "Electric Blue",
                "Neon Green",
                "Golden Yellow",
                "Hot Red",
                "White Glow",
                "Custom Color",
            ],
            index=0,
        )
        custom_color = st.color_picker("Custom Color", "#00D5FF")
        trail_color = custom_color if color_choice == "Custom Color" else TRAIL_COLORS[color_choice]
        line_thickness = st.slider("Trail line thickness", min_value=1, max_value=12, value=4)
        glow_effect = st.toggle("Glow effect", value=True)
        show_points = st.toggle("Show trajectory points", value=True)
        show_markers = st.toggle("Show start and end markers", value=True)

    analysis_settings = {
        "fps": int(fps),
        "phone_distance_m": float(phone_distance_m),
        "reference_length_m": float(reference_length_m),
        "reference_pixels": int(reference_pixels),
        "club_type": club_type,
        "camera_angle": camera_angle,
    }
    trail_settings = {
        "color_hex": trail_color,
        "line_thickness": int(line_thickness),
        "glow_effect": bool(glow_effect),
        "show_points": bool(show_points),
        "show_markers": bool(show_markers),
    }
    return analysis_settings, trail_settings


def progress_noop(_: float, __: str) -> None:
    return None


def scan_video_candidates(
    video_path: Path,
    max_frames: Optional[int],
    progress_callback: Callable[[float, str], None],
) -> tuple[List[List[Detection]], int]:
    detector = BallDetector()
    candidate_frames: List[List[Detection]] = []
    previous_frame = None
    processed = 0

    for frame_index, frame in iter_video_frames(video_path, max_frames=max_frames):
        candidates = detector.detect_candidates(frame, frame_index, previous_frame=previous_frame)
        candidate_frames.append(candidates)
        previous_frame = frame
        processed += 1

        if max_frames and processed % 10 == 0:
            progress = min(0.62, (processed / max_frames) * 0.62)
            progress_callback(progress, f"Scanning frame {processed} of {max_frames}")

    return candidate_frames, processed


def build_report(
    metadata: Dict[str, object],
    settings: Dict[str, object],
    trail_settings: Dict[str, object],
    points: List[TrajectoryPoint],
    launch_angle_deg: Optional[float],
    speed_mps: Optional[float],
    carry_m: Optional[float],
    confidence: float,
    overlay_path: Path,
    plot_path: Path,
) -> Dict[str, object]:
    speed_mph = speed_mps * 2.23693629 if speed_mps is not None else None
    speed_kmh = speed_mps * 3.6 if speed_mps is not None else None
    carry_yd = carry_m * 1.0936133 if carry_m is not None else None

    return {
        "product": "Golf Shot Vision",
        "estimated": True,
        "accuracy_notice": (
            "Estimated values from 2D video and OpenCV rule-based detection. "
            "This is not launch monitor accuracy."
        ),
        "video_metadata": metadata,
        "analysis_settings": settings,
        "trail_settings": trail_settings,
        "metrics": {
            "launch_angle_deg_estimated": launch_angle_deg,
            "initial_ball_speed_mps_estimated": speed_mps,
            "initial_ball_speed_mph_estimated": speed_mph,
            "initial_ball_speed_kmh_estimated": speed_kmh,
            "carry_m_estimated": carry_m,
            "carry_yd_estimated": carry_yd,
            "confidence_percent": confidence,
        },
        "output_files": {
            "overlay_video": str(overlay_path),
            "trajectory_plot": str(plot_path),
        },
        "trajectory": [asdict(point) for point in points],
    }


def analyze_video_file(
    video_path: Path,
    analysis_settings: Dict[str, object],
    trail_settings: Dict[str, object],
    progress_callback: Callable[[float, str], None] = progress_noop,
) -> Dict[str, object]:
    metadata_obj = get_video_metadata(video_path)
    metadata = asdict(metadata_obj)

    configured_fps = int(analysis_settings["fps"])
    fps = configured_fps if configured_fps > 0 else int(round(metadata_obj.fps or 30))
    if metadata_obj.frame_count > 0:
        max_frames = min(metadata_obj.frame_count, max(180, int(fps * 8)))
    else:
        max_frames = max(180, int(fps * 8))

    progress_callback(0.04, "Opening video and preparing frame scan")
    candidate_frames, processed_frames = scan_video_candidates(video_path, max_frames, progress_callback)
    progress_callback(0.68, "Linking bright moving targets into a ball track")

    raw_detections = select_best_trajectory(
        candidate_frames,
        frame_width=metadata_obj.width,
        frame_height=metadata_obj.height,
        min_points=5,
    )

    if len(raw_detections) < 5:
        return {
            "detected": False,
            "message": (
                "No reliable golf ball flight was detected. Try an iPhone slow-motion side-view clip "
                "with good light, a stable phone, and the ball visible against a clean background."
            ),
            "metadata": metadata,
            "processed_frames": processed_frames,
        }

    progress_callback(0.74, "Smoothing trajectory")
    points = smooth_trajectory(raw_detections)
    pixel_to_meter = float(analysis_settings["reference_length_m"]) / max(
        1.0,
        float(analysis_settings["reference_pixels"]),
    )

    launch_angle_deg = estimate_launch_angle(points, camera_angle=str(analysis_settings["camera_angle"]))
    speed_mps = estimate_initial_ball_speed(points, fps=fps, pixel_to_meter=pixel_to_meter)
    carry_m = estimate_carry(
        speed_mps=speed_mps,
        launch_angle_deg=launch_angle_deg,
        club_type=str(analysis_settings["club_type"]),
        camera_angle=str(analysis_settings["camera_angle"]),
    )
    confidence = compute_confidence(
        points,
        raw_detections,
        camera_angle=str(analysis_settings["camera_angle"]),
        processed_frames=processed_frames,
    )

    job_id = uuid.uuid4().hex[:10]
    overlay_path = TRACE_DIR / f"gsv_trace_{job_id}.mp4"
    plot_path = PLOT_DIR / f"gsv_trajectory_{job_id}.png"
    report_path = REPORT_DIR / f"gsv_report_{job_id}.json"

    progress_callback(0.82, "Rendering trajectory overlay video")
    render_traced_video(
        input_path=video_path,
        output_path=overlay_path,
        trajectory=points,
        color_bgr=hex_to_bgr(str(trail_settings["color_hex"])),
        thickness=int(trail_settings["line_thickness"]),
        glow=bool(trail_settings["glow_effect"]),
        show_points=bool(trail_settings["show_points"]),
        show_markers=bool(trail_settings["show_markers"]),
        max_frames=max_frames,
    )

    progress_callback(0.94, "Saving trajectory plot and report")
    save_trajectory_plot(points, plot_path, pixel_to_meter=pixel_to_meter)

    report = build_report(
        metadata=metadata,
        settings={**analysis_settings, "effective_fps": fps},
        trail_settings=trail_settings,
        points=points,
        launch_angle_deg=launch_angle_deg,
        speed_mps=speed_mps,
        carry_m=carry_m,
        confidence=confidence,
        overlay_path=overlay_path,
        plot_path=plot_path,
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    progress_callback(1.0, "Analysis complete")

    return {
        "detected": True,
        "report": report,
        "report_path": str(report_path),
        "overlay_path": str(overlay_path),
        "plot_path": str(plot_path),
        "original_path": str(video_path),
    }


def format_number(value: Optional[float], suffix: str, decimals: int = 1) -> str:
    if value is None or not math.isfinite(value):
        return "Estimated N/A"
    return f"Estimated {value:.{decimals}f}{suffix}"


def display_results(result: Dict[str, object]) -> None:
    if not result.get("detected"):
        st.warning(str(result.get("message", "No reliable ball flight was detected.")))
        return

    report = result["report"]
    metrics = report["metrics"]
    launch_angle = metrics["launch_angle_deg_estimated"]
    speed_mph = metrics["initial_ball_speed_mph_estimated"]
    speed_mps = metrics["initial_ball_speed_mps_estimated"]
    carry_m = metrics["carry_m_estimated"]
    carry_yd = metrics["carry_yd_estimated"]
    confidence = metrics["confidence_percent"]

    st.markdown("### Estimated Results")
    st.caption("Estimated from 2D video. Not professional launch monitor accuracy.")
    st.markdown(
        f"""
        <div class="metric-grid">
            {metric_card("Launch Angle", format_number(launch_angle, " deg"), "Estimated takeoff angle")}
            {metric_card("Estimated Ball Speed", format_number(speed_mph, " mph"), format_number(speed_mps, " m/s"))}
            {metric_card("Estimated Carry", format_number(carry_m, " m"), format_number(carry_yd, " yd"))}
            {metric_card("Confidence", f"{confidence:.0f}%" if confidence is not None else "N/A", "Rule-based track quality")}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Video Output")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Original video")
        st.video(str(result["original_path"]))
    with right:
        st.markdown("#### Traced video")
        st.video(str(result["overlay_path"]))

    st.markdown("### Trajectory Curve")
    st.image(str(result["plot_path"]), use_container_width=True)

    overlay_bytes = Path(str(result["overlay_path"])).read_bytes()
    report_bytes = json.dumps(report, indent=2).encode("utf-8")
    dl_left, dl_right = st.columns(2)
    with dl_left:
        st.download_button(
            "Download traced video",
            data=overlay_bytes,
            file_name=Path(str(result["overlay_path"])).name,
            mime="video/mp4",
            use_container_width=True,
        )
    with dl_right:
        st.download_button(
            "Download JSON analysis report",
            data=report_bytes,
            file_name=Path(str(result["report_path"])).name,
            mime="application/json",
            use_container_width=True,
        )


def run_analysis_ui(
    video_path: Path,
    analysis_settings: Dict[str, object],
    trail_settings: Dict[str, object],
    session_key: str,
) -> None:
    progress_bar = st.progress(0.0, text="Starting analysis")

    def update_progress(value: float, text: str) -> None:
        progress_bar.progress(min(1.0, max(0.0, value)), text=text)

    try:
        with st.spinner("Analyzing ball flight..."):
            result = analyze_video_file(video_path, analysis_settings, trail_settings, update_progress)
        st.session_state[session_key] = result
    except VideoLoadError as exc:
        st.error(f"Could not open the video: {exc}")
    except Exception as exc:
        st.error("Analysis failed, but the app is still running. Please try another clip or adjust settings.")
        with st.expander("Technical details"):
            st.code(str(exc))
    finally:
        progress_bar.empty()


def render_upload_tab(
    analysis_settings: Dict[str, object],
    trail_settings: Dict[str, object],
) -> None:
    st.markdown('<div class="tab-intro">Recommended mode: upload an iPhone slow-motion clip.</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload iPhone slow-motion video",
        type=["mp4", "mov", "m4v"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.info("Upload an mp4, mov, or m4v clip to begin.")
        return

    try:
        saved_path = save_uploaded_file(uploaded_file, UPLOAD_DIR)
        st.video(str(saved_path))
        analyze_clicked = st.button("Analyze uploaded video", type="primary", use_container_width=True)
        if analyze_clicked:
            run_analysis_ui(saved_path, analysis_settings, trail_settings, "upload_result")
    except Exception as exc:
        st.error("The uploaded file could not be saved or previewed.")
        with st.expander("Technical details"):
            st.code(str(exc))

    if "upload_result" in st.session_state:
        display_results(st.session_state["upload_result"])


def render_record_tab(
    analysis_settings: Dict[str, object],
    trail_settings: Dict[str, object],
) -> None:
    recorded_path = render_recorder_beta(st, RECORDING_DIR)
    if recorded_path:
        st.markdown("### Recorded clip")
        st.video(str(recorded_path))
        if st.button("Analyze recorded clip", type="primary", use_container_width=True):
            run_analysis_ui(Path(recorded_path), analysis_settings, trail_settings, "record_result")

    if "record_result" in st.session_state:
        display_results(st.session_state["record_result"])


def main() -> None:
    st.set_page_config(page_title="Golf Shot Vision", layout="wide")
    ensure_output_dirs([UPLOAD_DIR, TRACE_DIR, REPORT_DIR, PLOT_DIR, RECORDING_DIR])
    st.markdown(get_css(), unsafe_allow_html=True)

    render_hero()
    analysis_settings, trail_settings = sidebar_settings()

    upload_tab, record_tab = st.tabs(["Upload Video", "Record Video Beta"])
    with upload_tab:
        render_upload_tab(analysis_settings, trail_settings)
    with record_tab:
        render_record_tab(analysis_settings, trail_settings)


if __name__ == "__main__":
    main()
