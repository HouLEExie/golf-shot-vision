from __future__ import annotations

import json
import math
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd
import streamlit as st

from detection.roi_tracker import ROITracker, TrackerConfig
from physics.carry_estimator import estimate_carry
from physics.confidence import ConfidenceSummary, summarize_confidence
from physics.launch_angle import estimate_launch_angle
from physics.speed_estimator import estimate_initial_ball_speed
from styles import get_css
from tracking.trajectory_cleaner import (
    TrajectoryPoint,
    compute_confidence,
    smooth_trajectory,
)
from tracking.trajectory_validator import TrajectoryValidation, validate_trajectory
from video.loader import (
    VideoLoadError,
    ensure_output_dirs,
    get_video_metadata,
    save_uploaded_file,
)
from video.recorder import render_recorder_beta
from visualization.debug import save_debug_overlay_image
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
    "电光蓝": "#00D5FF",
    "荧光绿": "#39FF88",
    "金黄色": "#FFD23F",
    "亮红色": "#FF3B30",
    "白色光晕": "#F8FBFF",
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
            <h2>iPhone 慢动作高尔夫球路追踪器</h2>
            <p>上传 iPhone 慢动作视频，追踪球路轨迹，并估算起飞角、球速和飞行距离。</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def infer_video_quality_mode(fps: int) -> str:
    if fps < 60:
        return "低帧率模式：30fps"
    if fps == 60:
        return "普通模式：60fps"
    return "慢动作模式：120fps / 240fps"


def fps_guidance(fps: int) -> tuple[str, str]:
    if fps < 60:
        return (
            "warning",
            "当前视频帧率较低，高尔夫球飞行速度很快，轨迹识别和距离估算可能不准确。建议使用 iPhone 慢动作模式拍摄。",
        )
    if fps == 60:
        return (
            "info",
            "当前为普通帧率视频，可进行粗略分析。如需更稳定识别，建议使用 120fps 或 240fps 慢动作视频。",
        )
    return ("success", "当前为慢动作视频，更适合高尔夫球轨迹识别。")


def parse_roi(values: Dict[str, int]) -> Optional[tuple[int, int, int, int]]:
    x1 = int(values["roi_x1"])
    y1 = int(values["roi_y1"])
    x2 = int(values["roi_x2"])
    y2 = int(values["roi_y2"])
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def optional_start_position(start_x: int, start_y: int) -> tuple[Optional[float], Optional[float]]:
    if start_x <= 0 and start_y <= 0:
        return None, None
    return float(start_x), float(start_y)


def sidebar_settings() -> tuple[Dict[str, object], Dict[str, object]]:
    with st.sidebar:
        st.markdown("## 分析参数")
        recognition_mode = st.selectbox(
            "识别模式",
            ["半自动识别，推荐", "自动识别"],
            index=0,
            help="半自动模式会优先使用起始球位置和 ROI 区域，稳定性明显更好。",
        )
        fps = st.selectbox("视频帧率 FPS", [30, 60, 120, 240], index=3)
        inferred_quality_mode = infer_video_quality_mode(int(fps))
        video_quality_mode = st.selectbox(
            "视频质量模式",
            ["低帧率模式：30fps", "普通模式：60fps", "慢动作模式：120fps / 240fps"],
            index=["低帧率模式：30fps", "普通模式：60fps", "慢动作模式：120fps / 240fps"].index(inferred_quality_mode),
        )
        guidance_type, guidance_text = fps_guidance(int(fps))
        if guidance_type == "warning":
            st.warning(guidance_text)
        elif guidance_type == "info":
            st.info(guidance_text)
        else:
            st.success(guidance_text)

        phone_distance_m = st.number_input(
            "手机到球距离（米）",
            min_value=0.5,
            max_value=30.0,
            value=4.0,
            step=0.5,
        )
        reference_length_m = st.number_input(
            "参考物真实长度（米）",
            min_value=0.05,
            max_value=20.0,
            value=1.0,
            step=0.05,
        )
        reference_pixels = st.number_input(
            "参考物像素长度（px）",
            min_value=1,
            max_value=10000,
            value=300,
            step=10,
        )
        club_type = st.selectbox(
            "球杆类型",
            ["一号木", "球道木", "混合杆", "铁杆", "挖起杆"],
            index=0,
        )
        camera_angle = st.selectbox(
            "拍摄角度",
            ["侧面拍摄", "后方拍摄"],
            index=0,
        )
        ball_start_region = st.selectbox(
            "球起始搜索区域",
            ["画面下半部（推荐）", "左下区域", "中下区域", "右下区域", "全画面"],
            index=0,
            help="如果轨迹跑到人身上，请改成球所在的左下、中下或右下区域。全画面最宽松，也最容易误跟踪。",
        )
        st.caption("提示：单目视频无法像雷达一样精确测距。请用参考物校准比例，估算结果只适合训练反馈。")

        st.markdown("---")
        with st.expander("高级识别设置", expanded=True):
            start_frame = st.number_input("起始帧 start_frame", min_value=0, max_value=20000, value=0, step=1)
            analyze_frames = st.number_input("分析帧数 analyze_frames", min_value=10, max_value=2000, value=80, step=10)
            start_x = st.number_input("球起始位置 start_x（px）", min_value=0, max_value=10000, value=0, step=1)
            start_y = st.number_input("球起始位置 start_y（px）", min_value=0, max_value=10000, value=0, step=1)
            flight_direction = st.selectbox("飞行方向", ["左到右", "右到左", "向上"], index=0)

            st.caption("搜索区域 ROI：如果不填写或 x2/y2 小于等于 x1/y1，则使用全画面。")
            roi_x1 = st.number_input("roi_x1", min_value=0, max_value=10000, value=0, step=1)
            roi_y1 = st.number_input("roi_y1", min_value=0, max_value=10000, value=0, step=1)
            roi_x2 = st.number_input("roi_x2", min_value=0, max_value=10000, value=0, step=1)
            roi_y2 = st.number_input("roi_y2", min_value=0, max_value=10000, value=0, step=1)
            roi = parse_roi({"roi_x1": roi_x1, "roi_y1": roi_y1, "roi_x2": roi_x2, "roi_y2": roi_y2})
            if roi is None:
                st.warning("当前使用全画面识别，容易误识别反光点或背景亮点。建议使用半自动识别模式并设置搜索区域。")

            default_max_step = 190 if int(fps) < 60 else 125 if int(fps) == 60 else 72
            default_min_step = 3 if int(fps) < 60 else 2
            max_step_px = st.number_input("最大单帧位移 max_step_px", min_value=5, max_value=1000, value=default_max_step, step=5)
            min_step_px = st.number_input("最小单帧位移 min_step_px", min_value=0, max_value=200, value=default_min_step, step=1)
            smoothness = st.slider("轨迹平滑强度 smoothness", min_value=0.0, max_value=1.0, value=0.45, step=0.05)
            debug_mode = st.toggle("开启调试模式 debug mode", value=False)

            if recognition_mode.startswith("半自动") and (start_x <= 0 and start_y <= 0):
                st.warning("半自动识别建议填写 start_x 和 start_y，否则第一帧仍需在 ROI 内自动寻找球。")

        st.markdown("---")
        st.markdown("## 轨迹设置")
        color_choice = st.selectbox(
            "轨迹颜色",
            [
                "电光蓝",
                "荧光绿",
                "金黄色",
                "亮红色",
                "白色光晕",
                "自定义颜色",
            ],
            index=0,
        )
        custom_color = st.color_picker("自定义颜色", "#00D5FF")
        trail_color = custom_color if color_choice == "自定义颜色" else TRAIL_COLORS[color_choice]
        line_thickness = st.slider("轨迹粗细", min_value=1, max_value=12, value=4)
        glow_effect = st.toggle("发光效果", value=True)
        show_points = st.toggle("显示轨迹点", value=True)
        show_markers = st.toggle("显示起点和终点", value=True)

    analysis_settings = {
        "fps": int(fps),
        "video_quality_mode": video_quality_mode,
        "recognition_mode": recognition_mode,
        "phone_distance_m": float(phone_distance_m),
        "reference_length_m": float(reference_length_m),
        "reference_pixels": int(reference_pixels),
        "club_type": club_type,
        "camera_angle": camera_angle,
        "ball_start_region": ball_start_region,
        "start_frame": int(start_frame),
        "analyze_frames": int(analyze_frames),
        "start_x": int(start_x),
        "start_y": int(start_y),
        "flight_direction": flight_direction,
        "roi": roi,
        "roi_x1": int(roi_x1),
        "roi_y1": int(roi_y1),
        "roi_x2": int(roi_x2),
        "roi_y2": int(roi_y2),
        "max_step_px": float(max_step_px),
        "min_step_px": float(min_step_px),
        "smoothness": float(smoothness),
        "debug_mode": bool(debug_mode),
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


def build_report(
    metadata: Dict[str, object],
    settings: Dict[str, object],
    trail_settings: Dict[str, object],
    points: List[TrajectoryPoint],
    launch_angle_deg: Optional[float],
    speed_mps: Optional[float],
    carry_m: Optional[float],
    confidence_summary: ConfidenceSummary,
    validation: TrajectoryValidation,
    tracker_warnings: List[str],
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
            "结果基于 2D 视频和 OpenCV 规则检测估算，适合训练反馈，"
            "不等同于专业雷达设备数据。"
        ),
        "video_metadata": metadata,
        "analysis_settings": settings,
        "trail_settings": trail_settings,
        "recognition": {
            "video_fps": settings.get("effective_fps", settings.get("fps")),
            "video_quality_mode": settings.get("video_quality_mode"),
            "recognition_mode": settings.get("recognition_mode"),
            "confidence_level": confidence_summary.level,
            "confidence_reasons": confidence_summary.reasons,
            "validation_reasons": validation.reasons,
            "tracker_warnings": tracker_warnings,
            "should_retake_slow_motion": confidence_summary.should_retake_slow_motion,
        },
        "metrics": {
            "launch_angle_deg_estimated": launch_angle_deg,
            "initial_ball_speed_mps_estimated": speed_mps,
            "initial_ball_speed_mph_estimated": speed_mph,
            "initial_ball_speed_kmh_estimated": speed_kmh,
            "carry_m_estimated": carry_m,
            "carry_yd_estimated": carry_yd,
            "confidence_percent": confidence_summary.percent,
            "confidence_level": confidence_summary.level,
            "distance_reliable": confidence_summary.distance_reliable,
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
    start_x, start_y = optional_start_position(int(analysis_settings["start_x"]), int(analysis_settings["start_y"]))
    tracker_config = TrackerConfig(
        recognition_mode=str(analysis_settings["recognition_mode"]),
        video_quality_mode=str(analysis_settings["video_quality_mode"]),
        fps=fps,
        start_frame=int(analysis_settings["start_frame"]),
        analyze_frames=int(analysis_settings["analyze_frames"]),
        start_x=start_x,
        start_y=start_y,
        flight_direction=str(analysis_settings["flight_direction"]),
        roi=analysis_settings["roi"],
        max_step_px=float(analysis_settings["max_step_px"]),
        min_step_px=float(analysis_settings["min_step_px"]),
        smoothness=float(analysis_settings["smoothness"]),
        debug_mode=bool(analysis_settings["debug_mode"]),
    )

    progress_callback(0.08, "正在按 ROI 和预测窗口追踪球")
    tracking_result = ROITracker(tracker_config).track(video_path)
    raw_detections = tracking_result.detections
    processed_frames = len(tracking_result.candidate_counts)
    debug_payload = {
        "debug_rows": tracking_result.debug_rows,
        "candidate_counts": tracking_result.candidate_counts,
        "tracker_warnings": tracking_result.warnings,
        "stopped_reason": tracking_result.stopped_reason,
        "roi_used": tracking_result.roi_used,
    }

    if len(raw_detections) < 5:
        message = "当前视频未能稳定识别高尔夫球轨迹。建议使用半自动识别模式，手动设置起始帧、起始球位置和搜索区域。"
        if fps < 60:
            message = "当前视频帧率较低，球飞行过程中帧间位移过大，系统无法稳定追踪。建议使用 iPhone 慢动作 120fps 或 240fps 重新拍摄。"
        return {
            "detected": False,
            "message": message,
            "metadata": metadata,
            "processed_frames": processed_frames,
            **debug_payload,
        }

    progress_callback(0.74, "正在平滑轨迹")
    points = smooth_trajectory(raw_detections)
    validation = validate_trajectory(
        points,
        fps=fps,
        flight_direction=str(analysis_settings["flight_direction"]),
        min_step_px=float(analysis_settings["min_step_px"]),
        max_step_px=float(analysis_settings["max_step_px"]),
    )
    if not validation.is_valid:
        return {
            "detected": False,
            "message": "当前视频未能稳定识别高尔夫球轨迹。建议使用半自动识别模式，手动设置起始帧、起始球位置和搜索区域。",
            "metadata": metadata,
            "processed_frames": processed_frames,
            "validation_reasons": validation.reasons,
            "validation_warnings": validation.warnings,
            **debug_payload,
        }

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
    confidence_summary = summarize_confidence(
        base_confidence=confidence,
        validation_multiplier=validation.confidence_multiplier,
        fps=fps,
        point_count=len(points),
        recognition_mode=str(analysis_settings["recognition_mode"]),
        validation_reasons=validation.reasons,
    )
    if not confidence_summary.distance_reliable:
        speed_mps = None
        carry_m = None

    job_id = uuid.uuid4().hex[:10]
    overlay_path = TRACE_DIR / f"gsv_trace_{job_id}.mp4"
    plot_path = PLOT_DIR / f"gsv_trajectory_{job_id}.png"
    report_path = REPORT_DIR / f"gsv_report_{job_id}.json"
    debug_image_path = PLOT_DIR / f"gsv_debug_{job_id}.png"
    max_output_frames = int(analysis_settings["start_frame"]) + int(analysis_settings["analyze_frames"])

    progress_callback(0.82, "正在渲染轨迹视频")
    render_traced_video(
        input_path=video_path,
        output_path=overlay_path,
        trajectory=points,
        color_bgr=hex_to_bgr(str(trail_settings["color_hex"])),
        thickness=int(trail_settings["line_thickness"]),
        glow=bool(trail_settings["glow_effect"]),
        show_points=bool(trail_settings["show_points"]),
        show_markers=bool(trail_settings["show_markers"]),
        max_frames=max_output_frames,
    )

    progress_callback(0.94, "正在保存轨迹图和分析报告")
    save_trajectory_plot(points, plot_path, pixel_to_meter=pixel_to_meter)
    debug_overlay_path = None
    if bool(analysis_settings["debug_mode"]):
        debug_overlay_path = save_debug_overlay_image(
            video_path=video_path,
            output_path=debug_image_path,
            debug_rows=tracking_result.debug_rows,
            roi=tracking_result.roi_used,
            start_frame=int(analysis_settings["start_frame"]),
        )

    report = build_report(
        metadata=metadata,
        settings={**analysis_settings, "effective_fps": fps},
        trail_settings=trail_settings,
        points=points,
        launch_angle_deg=launch_angle_deg,
        speed_mps=speed_mps,
        carry_m=carry_m,
        confidence_summary=confidence_summary,
        validation=validation,
        tracker_warnings=tracking_result.warnings + validation.warnings,
        overlay_path=overlay_path,
        plot_path=plot_path,
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    progress_callback(1.0, "分析完成")

    return {
        "detected": True,
        "report": report,
        "report_path": str(report_path),
        "overlay_path": str(overlay_path),
        "plot_path": str(plot_path),
        "original_path": str(video_path),
        "confidence_summary": asdict(confidence_summary),
        "validation_reasons": validation.reasons,
        "validation_warnings": validation.warnings,
        "debug_image_path": str(debug_overlay_path) if debug_overlay_path else None,
        **debug_payload,
    }


def format_number(value: Optional[float], suffix: str, decimals: int = 1) -> str:
    if value is None or not math.isfinite(value):
        return "暂无估算"
    return f"约 {value:.{decimals}f}{suffix}"


def render_debug_section(result: Dict[str, object]) -> None:
    debug_rows = result.get("debug_rows") or []
    candidate_counts = result.get("candidate_counts") or []
    debug_image_path = result.get("debug_image_path")

    if not debug_rows and not candidate_counts and not debug_image_path:
        return

    with st.expander("调试输出", expanded=False):
        if debug_image_path:
            st.markdown("#### 候选点调试图")
            st.image(str(debug_image_path), use_container_width=True)

        if candidate_counts:
            st.markdown("#### 每帧候选点数量")
            counts_df = pd.DataFrame(candidate_counts)
            st.dataframe(counts_df, use_container_width=True)

        if debug_rows:
            st.markdown("#### 候选点评分明细")
            debug_df = pd.DataFrame(debug_rows)
            st.dataframe(debug_df, use_container_width=True)
            st.download_button(
                "下载轨迹点 debug CSV",
                data=debug_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="golf-shot-vision-debug.csv",
                mime="text/csv",
                use_container_width=True,
            )


def display_results(result: Dict[str, object]) -> None:
    if not result.get("detected"):
        st.warning(str(result.get("message", "没有检测到稳定的球路轨迹。")))
        for warning in result.get("tracker_warnings", []) or []:
            st.warning(str(warning))
        for warning in result.get("validation_warnings", []) or []:
            st.warning(str(warning))
        render_debug_section(result)
        return

    report = result["report"]
    metrics = report["metrics"]
    recognition = report.get("recognition", {})
    launch_angle = metrics["launch_angle_deg_estimated"]
    speed_mph = metrics["initial_ball_speed_mph_estimated"]
    speed_mps = metrics["initial_ball_speed_mps_estimated"]
    carry_m = metrics["carry_m_estimated"]
    carry_yd = metrics["carry_yd_estimated"]
    confidence = metrics["confidence_percent"]
    confidence_level = metrics.get("confidence_level", "未知")

    st.markdown("### 分析结果")
    st.caption("结果来自 2D 视频估算，不等同于专业雷达设备数据。")
    fps_value = int(report["analysis_settings"]["fps"])
    if fps_value < 60:
        st.warning("30fps 视频仅适合粗略参考，不建议用于准确距离估算。")
        speed_detail = "低可信度，仅供参考"
        carry_detail = "低可信度，仅供参考"
    elif fps_value == 60:
        speed_detail = "粗略估算"
        carry_detail = "粗略估算"
    else:
        speed_detail = format_number(speed_mps, " m/s")
        carry_detail = format_number(carry_yd, " 码")
    for warning in recognition.get("tracker_warnings", []) or []:
        st.warning(str(warning))
    if recognition.get("should_retake_slow_motion"):
        st.warning("建议使用 iPhone 慢动作 120fps 或 240fps 重新拍摄，以获得更稳定的轨迹。")
    st.info("识别置信度原因：" + "；".join(str(item) for item in recognition.get("confidence_reasons", [])))
    st.markdown(
        f"""
        <div class="metric-grid">
            {metric_card("起飞角", format_number(launch_angle, "°"), "估算出球角度")}
            {metric_card("估算球速", format_number(speed_mph, " mph"), speed_detail)}
            {metric_card("估算飞行距离", format_number(carry_m, " 米"), carry_detail)}
            {metric_card("识别置信度", f"{confidence:.0f}%" if confidence is not None else "暂无数据", f"等级：{confidence_level}")}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 视频结果")
    left, right = st.columns(2)
    with left:
        st.markdown("#### 原视频")
        st.video(str(result["original_path"]))
    with right:
        st.markdown("#### 轨迹视频")
        st.video(str(result["overlay_path"]))

    st.markdown("### 轨迹图表")
    st.image(str(result["plot_path"]), use_container_width=True)

    overlay_bytes = Path(str(result["overlay_path"])).read_bytes()
    report_bytes = json.dumps(report, indent=2).encode("utf-8")
    dl_left, dl_right = st.columns(2)
    with dl_left:
        st.download_button(
            "下载轨迹视频",
            data=overlay_bytes,
            file_name=Path(str(result["overlay_path"])).name,
            mime="video/mp4",
            use_container_width=True,
        )
    with dl_right:
        st.download_button(
            "下载分析报告",
            data=report_bytes,
            file_name=Path(str(result["report_path"])).name,
            mime="application/json",
            use_container_width=True,
        )

    render_debug_section(result)


def run_analysis_ui(
    video_path: Path,
    analysis_settings: Dict[str, object],
    trail_settings: Dict[str, object],
    session_key: str,
) -> None:
    progress_bar = st.progress(0.0, text="正在开始分析")

    def update_progress(value: float, text: str) -> None:
        progress_bar.progress(min(1.0, max(0.0, value)), text=text)

    try:
        with st.spinner("正在分析球路..."):
            result = analyze_video_file(video_path, analysis_settings, trail_settings, update_progress)
        st.session_state[session_key] = result
    except VideoLoadError as exc:
        st.error(f"无法打开视频：{exc}")
    except Exception as exc:
        st.error("检测失败：分析过程中出现异常。请更换视频或调整参数后重试。")
        with st.expander("技术详情"):
            st.code(str(exc))
    finally:
        progress_bar.empty()


def render_upload_tab(
    analysis_settings: Dict[str, object],
    trail_settings: Dict[str, object],
) -> None:
    st.markdown('<div class="tab-intro">推荐模式：上传 iPhone 慢动作视频。</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "上传 iPhone 慢动作视频",
        type=["mp4", "mov", "m4v"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.info("请上传 mp4、mov 或 m4v 视频开始分析。")
        return

    try:
        saved_path = save_uploaded_file(uploaded_file, UPLOAD_DIR)
        st.video(str(saved_path))
        analyze_clicked = st.button("开始分析", type="primary", use_container_width=True)
        if analyze_clicked:
            run_analysis_ui(saved_path, analysis_settings, trail_settings, "upload_result")
    except Exception as exc:
        st.error("上传失败：无法保存或预览该视频文件。")
        with st.expander("技术详情"):
            st.code(str(exc))

    if "upload_result" in st.session_state:
        display_results(st.session_state["upload_result"])


def render_record_tab(
    analysis_settings: Dict[str, object],
    trail_settings: Dict[str, object],
) -> None:
    recorded_path = render_recorder_beta(st, RECORDING_DIR)
    if recorded_path:
        st.markdown("### 录制视频")
        st.video(str(recorded_path))
        if st.button("分析录制视频", type="primary", use_container_width=True):
            run_analysis_ui(Path(recorded_path), analysis_settings, trail_settings, "record_result")

    if "record_result" in st.session_state:
        display_results(st.session_state["record_result"])


def main() -> None:
    st.set_page_config(page_title="Golf Shot Vision", layout="wide")
    ensure_output_dirs([UPLOAD_DIR, TRACE_DIR, REPORT_DIR, PLOT_DIR, RECORDING_DIR])
    st.markdown(get_css(), unsafe_allow_html=True)

    render_hero()
    analysis_settings, trail_settings = sidebar_settings()

    upload_tab, record_tab = st.tabs(["上传视频", "实时录制 Beta"])
    with upload_tab:
        render_upload_tab(analysis_settings, trail_settings)
    with record_tab:
        render_record_tab(analysis_settings, trail_settings)


if __name__ == "__main__":
    main()
