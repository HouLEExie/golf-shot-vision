# Golf Shot Vision

Golf Shot Vision is a Python + Streamlit prototype for tracing an iPhone slow-motion golf shot video and estimating launch angle, initial ball speed, carry distance, and confidence.

The first version uses OpenCV rule-based detection only. It does not use a deep learning model and should be treated as an estimated visual aid, not launch monitor accuracy.

## Features

- Upload iPhone slow-motion videos: `mp4`, `mov`, `m4v`
- Beta browser camera recording page with graceful fallback when `streamlit-webrtc` is unavailable
- OpenCV bright small-object and motion-continuity detection
- Kalman-style trajectory smoothing
- Custom trail color, thickness, glow, point markers, and start/end markers
- Traced output video, trajectory curve, and downloadable JSON report
- Modular structure for future YOLO or iOS app integration

## Project Structure

```text
golf-shot-vision/
|- app.py
|- styles.py
|- requirements.txt
|- README.md
|- video/
|  |- __init__.py
|  |- loader.py
|  `- recorder.py
|- detection/
|  |- __init__.py
|  |- ball_detector.py
|  `- motion_filter.py
|- tracking/
|  |- __init__.py
|  |- kalman_tracker.py
|  `- trajectory_cleaner.py
|- physics/
|  |- __init__.py
|  |- launch_angle.py
|  |- speed_estimator.py
|  `- carry_estimator.py
|- visualization/
|  |- __init__.py
|  |- overlay.py
|  `- plots.py
|- outputs/
`- sample_data/
```

## Install

Use Python 3.10 or newer on Windows.

```powershell
cd D:\CodexProjects\golf-shot-vision
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `streamlit-webrtc` or `av` has installation issues, the Upload Video mode can still be used after the core packages are installed. The Record Video Beta tab will show a friendly warning instead of crashing.

## Run

```powershell
streamlit run app.py
```

Then open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

## Shooting Recommendations

- Use iPhone slow motion, ideally 120 fps or 240 fps.
- Prefer side-view video for better launch angle and carry estimates.
- Keep the phone still on a tripod.
- Film in bright daylight or strong indoor light.
- Place the ball against a clean, high-contrast background.
- Keep the ball visible immediately after impact.
- Enter a real reference length and pixel length in the sidebar for better scale estimation.
- Avoid heavy camera shake, motion blur, and backgrounds with many bright moving objects.

## Notes on Estimates

The app estimates:

- Launch Angle
- Initial Ball Speed
- Carry Distance
- Detection Confidence

These values are labeled `Estimated` in the UI and JSON report. The calculations use 2D video, a reference pixel-to-meter scale, early-frame motion, and a simplified projectile model with rough drag factors. They are not professional launch monitor measurements.

## Future Extension Ideas

- Add YOLO or segmentation-based ball detection.
- Add manual impact-frame and reference-object calibration tools.
- Export frame-by-frame CSV.
- Add iOS-native capture and on-device model inference.
- Add cloud or local batch analysis.
