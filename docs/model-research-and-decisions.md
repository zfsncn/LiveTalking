# LiveTalking 调研分析与技术决策文档

> 记录时间：2026-10-12 ~ 2026-10-13
> 范围：官方文档研究、开源数字人模型全景调研、LatentSync / LivePortrait+JoyVASA 接入可行性、决策路线图
> 用途：团队学习与后续决策参考。所有实测数据基于 RTX 2080 Ti 22GB / Windows 11 + 原生 venv 环境。

---

## 1. 官方使用文档研究（doc.livetalking.ai/docs/usage/）

### 1.1 架构：三维正交组合

设计核心是「**数字人模型 × 传输方式 × TTS 引擎**」任意组合，默认 wav2lip + webrtc + edgetts。

| 维度 | 选项 | 关键信息 |
|---|---|---|
| 数字人模型（4） | wav2lip（默认）、musetalk、ernerf（独立分支）、Ultralight-Digital-Human | wav2lip 需 s3fd.pth + wav2lip256.pth + avatar 包；musetalk 生成 avatar 需 face_recognition/ffmpeg |
| 传输方式（4） | webrtc p2p（默认）、rtcpush→SRS、rtmp、virtualcam | webrtc 需 tcp:8010 + udp；推流需自建 SRS；virtualcam 接 OBS |
| TTS 引擎（8） | edgetts（默认）、gpt-sovits、xtts、腾讯云、豆包、阿里千问、azure、omnitts、mimotts（本项目新增） | 除 edgetts 外均需 key/自建服务；REF_FILE 在各引擎含义不同 |

5 项扩展能力：动作编排（customvideo_config.json）、LLM 对话（千问/OrcaRouter 网关）、多并发（--max_session）、语音输入（funasr 或浏览器原生）、虚拟摄像头。

### 1.2 文档滞后代码 3 处（2026-10 核实）

1. s3fd.pth 路径已从 `wav2lip/face_detection/...` 移至 `avatars/wav2lip/face_detection/detection/sfd/s3fd.pth`；
2. 文档未提及 `--config config.yaml`（优先级 CLI > YAML > 默认值）；
3. TTS 实现目录从 `ttsreal/` 重构为 `tts/`（九后端齐备）。

### 1.3 s3fd.pth 的真实影响

推理链路**不需要**它（wav2lip 运行时读预计算 `coords.pkl` + `full_imgs`，跳过人脸检测）；仅 genavatar 自制形象时触发加载——缺失时代码会从 `adrianbulat.com` 自动下载（国外站点，国内可能超时）。**建议预先放入**。

### 1.4 能力就绪度（对照部署）

| 能力 | 状态 |
|---|---|
| wav2lip + webrtc + mimotts | ✅ 全链路已验证（2026-10-13，当前主用模式） |
| LLM 对话 | ✅ `DASHSCOPE_API_KEY` 已填，qwen-plus |
| wav2lip + webrtc + edgetts | ✅ 全链路已验证（TTS 403 已补丁，降级备选） |
| 其他 TTS / 直播推流 / ASR 语音输入 / 虚拟摄像头 / 自制数字人 | ⚠️ 代码就绪，缺 key/服务/依赖 |

---

## 2. 开源数字人模型全景调研（2025–2026）

### 2.1 现有三个内置模型的技术路线

| 模型 | 路线 | 特点 |
|---|---|---|
| wav2lip | GAN 唇形修正 | 实时性最好，画质最基础 |
| MuseTalk | 隐空间扩散（VAE+U-Net） | 画质/同步率优于 wav2lip，仍实时 |
| Ultralight | 轻量 UNet | 单卡 5GB/1.5h 训练，牺牲画质 |

### 2.2 候选模型分三档

**第一档：实时/流式 + 消费级显卡（最值得集成）**

| 模型 | 出品方 | 实时性/显存 | 协议 |
|---|---|---|---|
| LivePortrait（+JoyVASA） | 快手 + 京东健康/浙大 | 60+ FPS，TensorRT 成熟 | **MIT（已核实）** |
| PersonaLive | 澳门大学/GVCLab（CVPR 2026） | 流式扩散，12GB 可跑 | **Apache-2.0** |
| LatentSync | 字节跳动 | 唇形同步，6GB | **Apache-2.0（已核实）** |
| Sonic | 腾讯+浙大 | 135ms 低延迟 | 需核实 |
| SoulX-FlashHead / LiveAct | Soul-AILab | 138 FPS@A100 | 权重已开放 |

**第二档：画质优先非实时**（EchoMimicV2 ~4.7k★ 半身手势、HunyuanVideo-Avatar 720p、InfiniteTalk/MultiTalk/LongCat-Video-Avatar-1.5 基于 Wan2.1-14B、Hallo3、FantasyTalking、SyncAnimation）——接入需缓冲适配，延迟 1–3 秒。

**第三档：端到端系统**（HeyGem/Duix、Linly-Talker、OpenTalking）——参考架构价值，不便拆模型引用。

### 2.3 接入机制

LiveTalking 的模型契约：新模型写一个 `BaseAvatar` 子类，实现 `inference_batch(index, audiofeat_batch)` + `paste_back_frame(res_frame, idx)` + genavatar 预处理。注册走映射表+动态 import，`--model` 直接可选。

---

## 3. 重点深度分析一：LatentSync

**定位：离线高质量唇形同步（V2V 批处理），不适合实时对话管线。**

- 链路：整段视频输入 → 逐帧 InsightFace 检测裁脸 → 16 帧窗口 × 20 步 DDIM 扩散 → mask 贴回 → ffmpeg 合成；
- 四大约束：① 延迟差 1–2 个数量级（0.64 秒视频渲染数秒，10 秒回复端到端 1–2 分钟）；② 1.6 版最低 18GB 显存；③ 输入要求完整视频；④ 依赖代差（torch 2.5.1+cu121 / diffusers 0.32 / insightface）；
- 利好：Apache-2.0；`audio2feature.py` 与 LiveTalking 的 MuseTalk 同源（Whisper tiny 384 维，可直接复用）；权重公开（HF `ByteDance/LatentSync-1.6`，走 hf-mirror）；
- 推荐接入方式：**独立本地服务/进程（与实时管线隔离）+ `/lipsync` 上传页面 + 共享目录交换文件**，做"视频配音/对口型"功能模块，实时管线零改动。

---

## 4. 重点深度分析二：LivePortrait + JoyVASA（当前最优选）

### 4.1 关键认知

LivePortrait 是纯渲染器（吃 keypoints 序列出帧），**单独进不了对话管线**；JoyVASA = LivePortrait 渲染器 + DiT 扩散音频前端（音频 → chinese-hubert 特征 → DiT 生成 100 帧 motion coef → keypoints → warp_decode 逐帧渲染 → paste_back）。**接 JoyVASA 即同时接上 LivePortrait**。两者 LICENSE 均为 MIT（已逐字核实），与 LiveTalking Apache-2.0 兼容。

### 4.2 BaseAvatar 契约映射（工作量约 5–7 天）

| 契约 | 实现 | 工作量 |
|---|---|---|
| `__init__` 加载模型 | LivePortraitWrapper（F/M/W/G/S）+ MotionGenerator | 直接移植 `src/modules/` |
| `self.asr` 音频特征 | 新写 `ChHubertASR(BaseASR)`（chinese-hubert-base，1024 维） | 半天 |
| `inference_batch` | motion 缓冲：攒满 100 帧（4 秒@25fps）触发一次扩散采样，之后逐帧 warp_decode | 1–2 天 |
| `paste_back_frame` | 复用 LivePortrait `utils/crop.py` 的 prepare_paste_back/paste_back | 半天 |
| genavatar | insightface 检测 → crop 256×256 → 存 box+mask | 1 天 |

### 4.3 实时性预算（2080Ti）

| 环节 | 开销 | 判定 |
|---|---|---|
| LivePortrait 单帧渲染（fp16） | ~10–20ms | ✅ 实时 |
| Motion generator（DiT 8 层，100 帧窗口） | 3–10 秒/次 | ⚠️ 首帧延迟 |
| 端到端 | 首句等几秒，之后每 4 秒音频产 4 秒视频连续输出 | 缓冲式准实时 |

对比 LatentSync（窗口 0.64 秒 vs JoyVASA 4 秒），延迟好一个数量级；motion 低频 + 渲染高频正是 ditto-talkinghead/AVTR-1 已验证实时系统的做法。

### 4.4 依赖兼容性（实测 venv torch 2.9.1+cu128 / numpy 2.5.2 / transformers 5.19）

| 依赖 | 结论 |
|---|---|
| torch/numpy/cv2/librosa（JoyVASA 推理核心 import 仅这些） | ✅ 全兼容 |
| diffusers/xformers（requirements 锁了但推理代码零 import） | ✅ 无冲突 |
| insightface + onnxruntime-gpu==1.18.0 | ⚠️ 需新增，兼容性好 |
| transformers 5.x ↔ chinese-hubert-base（override 了内部 API） | ⚠️ 唯一中风险点，先半天验证 |
| 显存（官方 8GB 4060 Laptop 可跑） | ✅ 22GB 富余 |

### 4.5 落地步骤

1. 本地 venv 内验证 transformers 5.19 能否加载 chinese-hubert-base 并跑通 JoyVASA hubert forward（半天，决定方案成立与否）；
2. hf-mirror 下载权重（~2GB：JoyVASA motion_generator + LivePortrait 全家桶 + insightface buffalo_l）；
3. 代码移植：`src/modules/` → `avatars/liveportrait/`；写 ChHubertASR + JoyVASAAvatar + genavatar；
4. 实测调优：采样步数/延迟/fp16/缓冲深度，可开 torch.compile（原生支持，提速 20–30%）；
5. 环境固化：把 edge-tts 补丁、新增依赖等写进安装脚本（可参考 `start.bat` / `tools/` 的组织方式），权重与形象文件落盘到位。

---

## 5. 决策汇总与路线图

### 5.1 已做决策

| 决策点 | 结论 | 理由 |
|---|---|---|
| 部署形态 | 原生 Windows venv（唯一形态） | 改码/调试高效；WebRTC 直连本机；无需 Docker/WSL2 |
| 镜像源策略 | torch 官方源，小件阿里云 | 阿里云 pytorch-wheels 限速 90KB/s |
| edge-tts | 7.2.8 + 端点补丁 | 微软 403，官方修复未发版 |
| LLM | dashscope qwen-plus（OrcaRouter 不用） | 免费/稳定；第三方网关过一手对话内容有隐私顾虑 |
| 传输 | webrtc p2p | 本机部署下最优；推流才引入 SRS |
| TTS 阶梯 | edgetts（免费）→ mimotts（MiMo 云端音色，已接入）→ gpt-sovits/xtts（克隆） | 按需升级 |

### 5.2 近期行动清单

- [x] 原生 Windows venv 部署（Python 3.12.13 + torch 2.9.1+cu128，2026-10-12）
- [x] `DASHSCOPE_API_KEY` 配置、LLM 对话生效
- [x] s3fd.pth 就位（自制形象能力补全）
- [x] MiMo-V2.5-TTS 后端接入 + 真实 key 端到端验证 + 切换为主用模式（2026-10-13）
- [x] 一键启动脚本 `start.bat`（自动清端口 + 弹浏览器，2026-10-13）
- [ ] JoyVASA 风险点验证（transformers 5.x × chinese-hubert-base）
- [ ] JoyVASA 接入为第四个模型（获批后 5–7 天）
- [ ] LatentSync 独立服务（离线配音功能，可选）

### 5.3 合规提醒

1. wav2lip 权重限**非商业研究用途**；
2. 数字人涉深度伪造风险，对外服务画面保留显式标识（现有代码左下角已带 LiveTalking 水印）；
3. 集成新模型前逐个核实 LICENSE 与 LiveTalking（Apache-2.0）的兼容性。
