# DesktopPetForLinux

把**任意视频**变成透明桌面宠物，悬浮在 Linux 桌面最上层播放。

Turn **any video** into a transparent desktop pet that floats on top of your Linux desktop.

> 与同类项目的核心区别：**无需绿幕、无需透明素材**。内置 AI 语义分割（U2Net）预处理步骤，自动从普通视频中抠出前景，任何一段人物/角色视频都可以直接用。

> The key difference from similar projects: **no green screen, no pre-made transparent assets required.** Built-in AI semantic segmentation (U2Net) automatically removes the background from any ordinary video — just point it at a clip and go.

![platform](https://img.shields.io/badge/platform-Linux%20%28X11%29-blue)
![python](https://img.shields.io/badge/python-3.12-blue)

## 效果

- 透明背景，无边框，始终置顶
- 透明区域鼠标可穿透，点到下方窗口/桌面
- 左键拖动移动位置，右键切换角色或退出
- 支持多个角色，右键菜单一键切换，记住上次选择

## Features

- Transparent background, borderless window, always on top
- Mouse clicks pass through transparent areas to windows beneath
- Left-drag to reposition, right-click to switch character or quit
- Multiple characters supported, switchable from the context menu, last selection remembered

## 依赖

**系统要求：** Linux Mint 22.3（或其他 X11 + 合成器的发行版）、ffmpeg

```bash
# 创建虚拟环境并安装依赖
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu
```

## Dependencies

**Requirements:** Linux Mint 22.3 (or any X11 distro with a compositor), ffmpeg

```bash
# Create virtual environment and install dependencies
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu
```

## 目录结构

```
desktop-dancer/
├── remove_bg.py      # 阶段1：AI 抠图，生成透明帧序列
├── dancer.py         # 阶段2：GTK 透明窗口播放
├── dancer/           # 角色目录
│   ├── .last         # 记录上次使用的角色（自动生成）
│   ├── anna01/       # 一个角色
│   │   ├── metadata.json
│   │   ├── frame_0001.png
│   │   └── ...
│   └── alyx2/        # 另一个角色
└── docs/
    └── backgroundremover.md
```

## Directory Structure

```
desktop-dancer/
├── remove_bg.py      # Phase 1: AI background removal, generates transparent frame sequence
├── dancer.py         # Phase 2: GTK transparent window player
├── dancer/           # Character directory
│   ├── .last         # Last used character (auto-generated)
│   ├── anna01/       # A character
│   │   ├── metadata.json
│   │   ├── frame_0001.png
│   │   └── ...
│   └── alyx2/        # Another character
└── docs/
    └── backgroundremover.md
```

## 使用

### 第一步：抠图

```bash
.venv/bin/python remove_bg.py --input 你的视频.mp4 --frames-dir dancer/角色名
```

首次运行会自动下载 U2Net 模型（约 176MB）。441 帧视频在 CPU 上约需 3~5 分钟。

抠图完成后，建议先预览一帧确认效果：

```bash
eog dancer/角色名/frame_0001.png
```

### 第二步：启动

```bash
.venv/bin/python dancer.py
```

常用参数：

| 参数 | 说明 | 默认 |
|------|------|------|
| `--dancer-dir` | 角色根目录 | `dancer` |
| `--scale` | 缩放比例 | `1.0` |
| `--sticky` | 在所有工作区显示 | 否 |
| `--x` / `--y` | 窗口初始位置 | 右下角 |

## Usage

### Step 1: Background Removal

```bash
.venv/bin/python remove_bg.py --input your_video.mp4 --frames-dir dancer/character_name
```

On first run, the U2Net model (~176MB) is downloaded automatically. A 441-frame video takes roughly 3–5 minutes on CPU.

After processing, preview a frame to verify the result:

```bash
eog dancer/character_name/frame_0001.png
```

### Step 2: Launch

```bash
.venv/bin/python dancer.py
```

Common options:

| Option | Description | Default |
|--------|-------------|---------|
| `--dancer-dir` | Character root directory | `dancer` |
| `--scale` | Display scale factor | `1.0` |
| `--sticky` | Show on all workspaces | off |
| `--x` / `--y` | Initial window position | bottom-right |

## 控制

| 操作 | 效果 |
|------|------|
| 左键拖动 | 移动窗口 |
| 右键单击 | 弹出菜单（切换角色 / 退出） |

## Controls

| Action | Effect |
|--------|--------|
| Left-drag | Move window |
| Right-click | Context menu (switch character / quit) |

## 技术说明

- **抠图**：[backgroundremover](https://github.com/nadermx/backgroundremover)（U2Net human_seg 模型）逐帧处理，输出 RGBA PNG 序列
- **展示**：GTK3 + RGBA visual 实现透明窗口，Cairo 逐帧绘制，GLib 定时器驱动动画
- **鼠标穿透**：`input_shape_combine_region` 按帧更新，透明像素不响应鼠标事件

## Technical Notes

- **Background removal**: [backgroundremover](https://github.com/nadermx/backgroundremover) (U2Net human_seg) processes each frame and outputs an RGBA PNG sequence
- **Display**: Transparent window via GTK3 RGBA visual, Cairo per-frame rendering, GLib timer-driven animation
- **Click-through**: `input_shape_combine_region` updated per frame so transparent pixels never intercept mouse events
