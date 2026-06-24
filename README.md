# Golf Shot Vision

iPhone 慢动作高尔夫球路追踪器。推荐使用 iPhone 原生相机慢动作拍摄后，再上传视频分析。

## 功能

- 上传 iPhone 慢动作视频：`mp4`、`mov`、`m4v`
- 生成带轨迹叠加的球路视频
- 显示起飞角、估算球速、估算飞行距离和识别置信度
- 支持轨迹颜色、轨迹粗细、发光效果、轨迹点、起点和终点设置
- 支持下载轨迹视频和 JSON 分析报告
- 实时录制 Beta：在 HTTPS 环境下更可能正常，但仍受手机浏览器限制

## 本地运行

安装依赖：

```powershell
pip install -r requirements.txt
```

启动应用：

```powershell
streamlit run app.py
```

电脑本机访问：

```text
http://localhost:8501
```

## 手机访问本地页面

如果使用局域网本地访问，电脑和手机必须连接同一个 Wi-Fi。

1. 在 Windows PowerShell 中运行：

```powershell
ipconfig
```

2. 找到当前 Wi-Fi 的 IPv4 地址，例如：

```text
192.168.1.8
```

3. 使用下面命令启动：

```powershell
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

4. 手机上打开：

```text
http://电脑IPv4地址:8501
```

例如：

```text
http://192.168.1.8:8501
```

5. 如果打不开，检查 Windows 防火墙是否允许 Python 或 Streamlit 访问专用网络。

## 公网部署到 Streamlit Community Cloud

部署到 Streamlit Community Cloud 后，手机不需要和电脑连接同一个 Wi-Fi。手机可以使用 4G、5G 或任意网络打开云端地址：

```text
https://xxx.streamlit.app
```

部署步骤：

1. 将本项目推送到 GitHub 仓库。
2. 打开 Streamlit Community Cloud，选择 New app。
3. 选择 GitHub 仓库和分支。
4. Main file path 填写：

```text
app.py
```

5. 点击 Deploy。Streamlit Cloud 会根据 `requirements.txt` 自动安装依赖。

云端部署说明：

- `requirements.txt` 使用 `opencv-python-headless`，避免云端缺少 `libGL.so.1` 导致 OpenCV 导入失败。
- `.streamlit/config.toml` 不绑定本地 IP 或固定端口，适合 Streamlit Community Cloud 自动分配运行环境。
- 上传 iPhone 慢动作视频是推荐模式。
- 实时录制 Beta 需要 HTTPS 环境。云端部署后摄像头权限更可能正常，但仍受 iPhone、Android 和具体手机浏览器限制。
- 云端文件系统是临时环境，上传的视频和生成的轨迹视频只适合本次会话下载，不建议当作长期存储。

## 实时录制说明

- 实时录制是 Beta 功能。
- 手机浏览器通过局域网 `http` 打开时，部分浏览器可能无法调用摄像头。
- 部分浏览器要求摄像头权限必须在 HTTPS 环境下才能使用。
- 第一版推荐用户使用 iPhone 原生相机慢动作拍摄后，再上传视频分析。

## 拍摄建议

- 优先使用侧面拍摄，并让球、球杆和出球方向尽量清晰。
- iPhone 慢动作建议使用 120 FPS 或 240 FPS。
- 保持手机稳定，画面亮度充足。
- 避免背景中有大量明亮移动物体。
- 如果轨迹跟踪到人身上，请在侧边栏把“球起始搜索区域”改成球所在的左下、中下或右下区域。
- 如果需要更准确的比例估算，请在画面中放入已知长度的参考物，并填写参考物真实长度和参考物像素长度。
- 如果上传 MOV 无法解码，可以在 iPhone 或电脑上导出为 H.264 MP4 后再试。

## 结果说明

本项目基于 2D 视频像素轨迹、参考物比例和简化弹道模型估算结果，适合训练反馈，不等同于专业雷达设备数据。
