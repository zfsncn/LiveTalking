# LiveTalking 本地部署复刻手册（原生 Windows venv）

> **文档目标**：本地电脑重装系统后，依据本文档从零复刻到当前项目的运行状态。
> **当前形态**：LiveTalking（原生 Windows venv，实时数字人）以 **模式 C（MiMo-V2.5-TTS）** 运行——TTS 走 MiMo API 云端音色「冰糖」，无需任何本地/容器化 TTS 服务；LLM 对话用阿里云百炼 qwen-plus。**全项目零 Docker 依赖**。
> **实测环境**：Windows 11 · RTX 2080 Ti 22GB · 驱动 591.86
> **记录时间**：2026-10-13　|　**总耗时**：约 1–1.5 小时（大头是依赖与模型下载）

---

## 0. 最终架构总览

```
┌─────────────────────────────────────────────────────────────┐
│  浏览器  http://localhost:8010/index.html                     │
│    ▲  WebRTC（音视频流，本机直连）                              │
│    │                                                          │
│  ┌─┴──────────────────────────────────────────────┐         │
│  │  LiveTalking（原生 Windows venv，监听 8010）      │         │
│  │  python app.py --model wav2lip --tts mimotts     │         │
│  └──────┬───────────────────────────┬──────────────┘         │
│         │ TTS（MiMo chat.completions）│ LLM（qwen-plus）        │
│         ▼                            ▼                        │
│  ┌──────────────────┐      ┌──────────────────────┐          │
│  │ MiMo-V2.5-TTS API │      │ 阿里云百炼 DashScope  │          │
│  │ api.xiaomimimo.com│      │ dashscope.aliyuncs.com│          │
│  │ 音色「冰糖」       │      │ （需 DASHSCOPE_API_KEY）│         │
│  └──────────────────┘      └──────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

### 组件与端口清单

| 组件                  | 形态        | 端口     | 位置               | 说明                              |
| ------------------- | --------- | ------ | ---------------- | ------------------------------- |
| **LiveTalking**     | 原生 venv   | `8010` | `D:\LiveTalking` | 实时数字人主服务（唯一本地组件）                 |
| MiMo-V2.5-TTS API   | 云端 API    | —      | 小米 MiMo           | 语音合成，音色「冰糖」（需 `MIMO_API_KEY`）   |
| DashScope LLM       | 云端 API    | —      | 阿里云              | qwen-plus 对话（可选，需 `DASHSCOPE_API_KEY`） |

### 两种运行模式（按需选择）

| 模式                                           | TTS 来源          | 启动参数                                                                 | 依赖                       |
| -------------------------------------------- | --------------- | -------------------------------------------------------------------- | ------------------------ |
| **C. MiMo-V2.5-TTS 云端音色（当前采用，已验证）**          | MiMo API        | `--tts mimotts --REF_FILE 冰糖 --MIMO_STYLE "自然温柔的女声，语速适中"`       | `.env` 配置 `MIMO_API_KEY` |
| **B. 免费云端语音（降级备选）**                          | edge-tts（微软）   | `--tts edgetts`                                                      | 需打 403 补丁（§2.4）          |

> **模式 C 说明（2026-10-13 实测）**：`tts/mimotts.py` 以 `@register("tts","mimotts")` 注册，走 MiMo chat.completions audio 模式（user=风格指令 / assistant=文本），返回 base64 WAV（24kHz）经 resampy 重采样至 16kHz、按 320 样本（20ms）分帧推流；API key 由 `app.py` 的 `load_dotenv()` 从 `.env` 读入 `MIMO_API_KEY`。完整启动命令见 §2.7。

---

## 1. 环境前置要求（重装系统后第一步）

| 项目             | 要求                                          | 验证命令                 |
| -------------- | ------------------------------------------- | -------------------- |
| **Windows**    | 10/11 x64                                    | —                    |
| **NVIDIA 驱动**  | ≥ R525（当前 591.86），支持 CUDA 12.x              | `nvidia-smi` 能列出 GPU |
| **Git**        | 任意新版                                        | `git --version`      |
| **磁盘空间**       | ≥ 20GB 空闲（venv + torch ~8GB、模型/形象 ~1GB）      | —                    |
| **网络**         | 可访问 download.pytorch.org、Google Drive（下模型）、api.xiaomimimo.com、dashscope.aliyuncs.com | —                    |

安装顺序：**显卡驱动 → Git → uv**（Python 由 uv 装，不碰系统 Python）。**无需 Docker Desktop / WSL2。**

---

## 2. 组件：LiveTalking 原生 Windows 部署（venv）

> 这是数字人主体，也是本项目**唯一**的本地组件。采用原生 venv 单一部署形态：WebRTC 天然绑定宿主机网络、改码调试直接生效、不依赖任何容器运行时。
> 工作目录：`D:\LiveTalking`

### 2.1 安装 uv + Python 3.12

本机系统 Python 是 3.13（部分依赖无 cp313-win 轮子），**必须用 3.12**。用 uv 管理，零污染：

```bash
# 装 uv（若已有任意 Python）
pip install uv -i https://mirrors.aliyun.com/pypi/simple/

# 用 uv 装 Python 3.12 并建 venv
uv python install 3.12
uv venv D:\LiveTalking\.venv --python 3.12
# 验证：D:\LiveTalking\.venv\Scripts\python.exe --version → Python 3.12.13
```

### 2.2 安装依赖

```bash
set UV=C:\path\to\uv.exe   # uv 可执行文件路径

# ① requirements 小件走阿里云源
%UV% pip install -r D:\LiveTalking\requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# ② torch 三件套走官方源（阿里云 pytorch-wheels 限速 90KB/s，切勿用！）
%UV% pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 ^
  --index-url https://download.pytorch.org/whl/cu128
# 约 2.5GB，7–10 分钟
```

> 用 uv 装时须设环境变量 `VIRTUAL_ENV=D:\LiveTalking\.venv`，或直接用 `.venv\Scripts\pip.exe`。

### 2.3 打 edge-tts 403 补丁（仅模式 B 需要）

微软 2026-10 起旧端点全量 403。**模式 C（MiMo）不受影响、可跳过本节**；仅当要用免费的 edge-tts（模式 B）时执行：

```python
# 在 .venv 里运行（干净进程，不要和 import edge_tts 同进程）
import pathlib, edge_tts
p = pathlib.Path(edge_tts.__file__).parent / 'constants.py'
s = p.read_text()
new = ('WSS_URL = (\n'
       '    "wss://api.msedgeservices.com/tts/cognitiveservices/websocket/v1"\n'
       '    f"?Ocp-Apim-Subscription-Key={TRUSTED_CLIENT_TOKEN}"\n)')
# 注意：新版 edge-tts 7.2.8 的格式是 WSS_URL = f"wss://{BASE_URL}/edge/v1?..."
s = s.replace('WSS_URL = f"wss://{BASE_URL}/edge/v1?TrustedClientToken={TRUSTED_CLIENT_TOKEN}"', new)
p.write_text(s)
```

> **坑**：① 令牌是 URL query 参数（`?Ocp-Apim-Subscription-Key=`），不是 header；② 打补丁后必须用**全新进程**测试（同进程内模块已加载旧常量会误报 403）；③ `pip install --reinstall edge-tts` 会冲掉补丁，需重打。

### 2.4 下载 LiveTalking 模型与形象

```bash
pip install gdown
python -m gdown --id 1wu6XujFL9rF-0P2l44G6kpeapeY0cME7 -O D:\LiveTalking\models\wav2lip.pth
# → models/wav2lip.pth（205MB）

python -m gdown --id 1aU-9SMEAZWN00hbvAGlRHG2iB17r9dEW -O D:\LiveTalking\data\avatars\wav2lip256_avatar1.tar.gz
cd D:\LiveTalking\data\avatars && tar -xzf wav2lip256_avatar1.tar.gz
# → data/avatars/wav2lip256_avatar1/{coords.pkl, face_imgs/, full_imgs/}
```

**（可选）s3fd.pth**：仅自制形象（genavatar / avatar.html）时需要，放入：

```
D:\LiveTalking\avatars\wav2lip\face_detection\detection\sfd\s3fd.pth   # 89MB
```

日常推理不需要它（wav2lip 运行时读预计算 coords.pkl，跳过人脸检测）。

### 2.5 配置 .env（LLM / MiMo key）

`D:\LiveTalking\.env`（从 `.env.example` 复制后填）：

```
DASHSCOPE_API_KEY=sk-xxxx      # 阿里云百炼，填了才有 LLM 对话；不填走 echo 复读
MIMO_API_KEY=sk-xxxx           # MiMo-V2.5-TTS（模式 C 必需），mimo.mi.com 申请，公测免费
TENCENT_APPID=xxxxxxxx          # 以下均可留占位符（用不到的 TTS 引擎）
TENCENT_SECRET_KEY=xxxxxxxx
TENCENT_SECRET_ID=xxxxxxxx
ORCAROUTER_API_KEY=xxxxxxxx
DOUBAO_API_KEY=xxxxxxxx
AZURE_SPEECH_KEY=xxxxxxxx
AZURE_TTS_ENDPOINT=xxxxxxxx
```

> `.env` 由 `python-dotenv` 自动加载（`app.py` 的 `load_dotenv()`），**无需手动 export，不要提交进 git**。

### 2.6 启动 LiveTalking（模式 C：MiMo-V2.5-TTS，主用）

```bash
cd D:\LiveTalking
.venv\Scripts\python.exe app.py ^
  --transport webrtc --model wav2lip --avatar_id wav2lip256_avatar1 ^
  --listenport 8010 ^
  --tts mimotts --REF_FILE 冰糖 --MIMO_STYLE "自然温柔的女声，语速适中"
```

启动成功标志（日志）：

```
INFO:utils.logger:Namespace(... tts='mimotts', REF_FILE='冰糖', MIMO_STYLE='自然温柔的女声，语速适中' ...)
INFO:utils.logger:Load checkpoint from: ./models/wav2lip.pth
INFO:utils.logger:warmup model...
INFO:utils.logger:start http server; http://<serverip>:8010
```

> **注意**：
> 1. 前置条件——`.env` 中已配置 `MIMO_API_KEY`（`load_dotenv()` 自动读入）。
> 2. **端口互斥**：重复启动前先停掉占用 8010 的旧进程——`netstat -ano | findstr 8010` 找到 PID，`taskkill //PID <pid> //F`，否则新进程报 `[WinError 10048]`（用 §2.9 的 start.bat 可自动清理）。
> 3. **2026-10-13 实测记录**：`curl http://localhost:8010/index.html` 返回 HTTP 200（20,733 字节）；MiMo 端到端合成验证通过——245,804 字节 WAV（24kHz/PCM_16/单声道/5.12s）、329 帧 × 320 样本、start/end 事件正确、端到端 4.86s。

### 2.7 模式 B（免费 edge-tts，降级备选）

```bash
.venv\Scripts\python.exe app.py --transport webrtc --model wav2lip ^
  --avatar_id wav2lip256_avatar1 --listenport 8010 --tts edgetts
```

> 需先打 §2.3 的 403 补丁；音质不及 MiMo，仅作无 key 备选。

### 2.8 换形象 / 换音色

- **换形象**：形象目录放 `data/avatars/<name>/`（标准三件套 coords.pkl + face_imgs + full_imgs），改 `--avatar_id <name>`。当前已有三个：`wav2lip256_avatar1`、`wav2lip_avatar_female_model`、`wav2lip_avatar_long_hair_girl`。也可在网页运行时用 `/human` 接口的 `avatar` 参数热切换。
- **模式 C 音色**：`--REF_FILE` 填 MiMo 音色名（8 个：冰糖/茉莉/苏打/白桦 + Mia/Chloe/Milo/Dean，当前用「冰糖」），风格用 `--MIMO_STYLE "..."` 描述（如「自然温柔的女声，语速适中」）。

### 2.9 一键启动（推荐日常使用）

双击项目根目录 **`start.bat`** 即可——自动清理 8010 占用 → 按模式 C 参数启动 → 就绪后自动弹出浏览器。底层由 `tools\start_livetalking.py` 实现，日志写入 `logs\start-<时间戳>.log`。

```bash
start.bat                                        # 模式 C 一键启动
start.bat --mode b                               # 切 edge-tts 免费模式
start.bat --avatar wav2lip_avatar_long_hair_girl # 换形象
start.bat --port 8011 --no-browser               # 换端口 / 不弹浏览器
```

---

## 3. 端到端验证清单

按顺序逐项验证，全绿即部署成功：

| # | 检查项              | 命令 / 操作                                                                       | 期望                        |
| - | ---------------- | ----------------------------------------------------------------------------- | ------------------------- |
| 1 | LiveTalking GPU  | `.venv\Scripts\python.exe -c "import torch;print(torch.cuda.is_available())"` | `True`                    |
| 2 | LiveTalking HTTP | `curl -o /dev/null -w "%{http_code}" http://localhost:8010/`                  | `302`                     |
| 3 | 页面可开             | 浏览器 `http://localhost:8010/index.html`                                        | 数字人界面                     |
| 4 | TTS 通路           | 独立跑 §附录 A 脚本                                                                  | 音频帧数 > 0 且含 start/end 事件   |
| 5 | LLM 对话           | 页面输入文字提交（需 DASHSCOPE key）                                                     | 回复非复读                     |
| 6 | **音视频实时**        | 页面点「开始连接」，输入文字                                                                | 数字人用「冰糖」音色实时说话 + 口型同步 |

### 附录 A：TTS 通路独立验证脚本（MiMo）

```python
# 在 D:\LiveTalking 下用 .venv 运行
import types
from dotenv import load_dotenv
load_dotenv()   # 读入 .env 的 MIMO_API_KEY

from tts.mimotts import MiMoTTS

class FrameCollector:
    """假装是 avatar，收下推流出来的音频帧。"""
    def __init__(self):
        self.frames = []
    def put_audio_frame(self, frame, event):
        self.frames.append((frame, event))

opt = types.SimpleNamespace(
    REF_FILE='冰糖', MIMO_STYLE='自然温柔的女声，语速适中',
    fps=25,   # base_tts 构造必需（决定 320 样本/块）
)
tts = MiMoTTS(opt, FrameCollector())
tts.txt_to_audio(('你好，这是通路测试。', {}))

frames = tts.parent.frames
print('音频帧数:', len(frames))
print('总样本数:', sum(len(f) for f, _ in frames))
print('事件序列:', [e.get('status') for _, e in frames])
# 帧数 > 0 且事件序列含 start/end 即通路正常（实测约 5 秒音频 → 250+ 帧）
```

---

## 4. 日常运维命令

```bash
# ── LiveTalking（原生 venv，模式 C）──
cd D:\LiveTalking
start.bat                       # 一键启动（推荐）：自动清端口 + 弹浏览器

# 或手动启动：
.venv\Scripts\python.exe app.py --transport webrtc --model wav2lip ^
  --avatar_id wav2lip256_avatar1 --listenport 8010 ^
  --tts mimotts --REF_FILE 冰糖 --MIMO_STYLE "自然温柔的女声，语速适中"

# Ctrl+C 停止；换端口改 --listenport
# 手动清理残留进程：netstat -ano | findstr 8010 → taskkill //PID <pid> //F
```

**启动顺序**：无依赖顺序问题——MiMo 与 DashScope 都是云端 API，LiveTalking 单进程起完即可用。

---

## 5. 故障排查速查表

| 现象                                    | 原因                            | 解法                                                                  |
| ------------------------------------- | ----------------------------- | ------------------------------------------------------------------- |
| 8010 端口被占（`[WinError 10048]`）         | 旧进程未退、重复启动                   | 用 `start.bat` 启动（自动清理）；或 `netstat -ano \| findstr 8010` 找 PID 后 `taskkill //PID <pid> //F` |
| MiMo 合成失败，日志 `MIMO_API_KEY 未设置`    | `.env` 未配或为空/占位符             | 填真实 key（`load_dotenv()` 自动加载，无需 export）                           |
| 页面能开但说话没声音                           | MiMo 调用异常或限流                 | 看 LiveTalking 日志有无 `MiMoTTS ... error`；公测限流 RPM 100，批量测试会 429       |
| edge-tts `WSServerHandshakeError 403` | 微软旧端点下线（模式 B）               | 打 §2.3 补丁，用新进程测                                                   |
| `torch` 下载极慢（~90KB/s）                 | 误用阿里云 pytorch-wheels          | 改官方源 `download.pytorch.org/whl/cu128`                               |
| `pip install` 依赖装不上                   | 系统 Python 3.13 缺 cp313-win 轮子 | 用 uv 的 Python 3.12                                                  |
| 数字人黑屏/连不上                             | WebRTC candidate 异常           | 确认用 `--transport webrtc`；本机访问无需额外网络配置                            |

---

## 6. 关键参数与版本速查

| 项                         | 值                                                       |
| ------------------------- | ------------------------------------------------------- |
| LiveTalking Python        | 3.12.13（uv 管理，`.venv`）                                  |
| LiveTalking torch         | 2.9.1+cu128                                             |
| LiveTalking edge-tts      | 7.2.8（仅模式 B 需打补丁）                                        |
| MiMo TTS                  | `mimo-v2.5-tts` @ `api.xiaomimimo.com/v1`；音色「冰糖」；24kHz→16kHz 重采样；320 样本/20ms 分帧；公测免费 RPM 100 |
| LLM                       | dashscope qwen-plus（`DASHSCOPE_API_KEY`）                 |
| GPU 架构注意                  | RTX 2080 Ti = Turing，**全程禁用 flash-attn，用 sdpa**         |

---

## 附：一键复刻检查清单

重装系统后，按此顺序执行：

- [ ] 装显卡驱动（≥R525）+ Git
- [ ] 装 uv + Python 3.12（§2.1）→ 装依赖（§2.2）
- [ ] （仅模式 B）打 edge-tts 补丁（§2.3）
- [ ] 下模型与形象（§2.4）→ 配 `.env`（§2.5，含 `MIMO_API_KEY`）
- [ ] 双击 `start.bat` 启动（§2.9）
- [ ] 跑 §3 验证清单，6 项全绿
- [ ] 打开 `http://localhost:8010/index.html` → 开始连接 → 输入文字 → 数字人用「冰糖」音色实时说话 ✅
