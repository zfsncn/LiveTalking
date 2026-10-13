# JoyVASA（LivePortrait 渲染器 + DiT 扩散运动生成）接入 LiveTalking 详细改造方案

> 方案 A：作为第四个 avatar 模型接入实时管线，`--model joyvasa` 直接可选。
> 部署形态：**原生 Windows venv**（D:\LiveTalking\.venv，Python 3.12，torch 2.9.1+cu128），无 Docker。
> 文档版本：2026-10-13，基于本地代码逐文件核实（所有行号均为当前工作区实际位置）。

---

## 0. 结论与总览

| 项 | 结论 |
|---|---|
| 可行性 | ✅ 可行。双 MIT 许可，与 Apache-2.0 兼容；依赖仅 2 项需新增、1 项需实测 |
| 接入方式 | 移植 JoyVASA `src/` 为 `avatars/joyvasa/` 子模块，新增 `joyvasa_avatar.py` + `ChHubertASR`（实为 PCM 缓冲 ASR）+ `genavatar.py` |
| 工作量 | 5-7 人日（风险验证 0.5 天 → 权重 0.5 天 → 移植 3-4 天 → 联调 1-2 天） |
| 实时形态 | 缓冲式准实时：首帧延迟 ≈ 一次扩散采样（估 3-10s，需实测），之后每 4 秒音频连续产 4 秒视频 |
| 画质 | 512×512 面部渲染（SPADE 上采样），显著高于 wav2lip 96 / musetalk 256 |
| 核心风险 | ① DDPM 完整 500 步采样的延迟 ② transformers 5.19 与 JoyVASA hubert 模块内部 API 兼容性 |

**一步到位的正确认知**：LivePortrait 是纯渲染器（无音频前端），单独接入无对话能力；JoyVASA = LivePortrait 全套渲染器 + DiT 运动生成器，接 JoyVASA 即同时获得 LivePortrait。

---

## 1. 已核实的代码级事实（方案依据）

### 1.1 JoyVASA 侧（D:\JoyVASA）

| 事实 | 位置 |
|---|---|
| `src/` 与 LivePortrait `src/` 同构，仅多 `dit_talking_head.py`、`hubert.py`、`wav2vec2.py`、`live_portrait_wmg_pipeline.py/_wrapper.py` | 目录对比 |
| 渲染契约：`prepare_source(img_crop_256) → get_kp_info → extract_feature_3d → transform_keypoint → warp_decode(f_s, x_s, x_d) → paste_back` | `live_portrait_wmg_wrapper.py:106-323`、`live_portrait_wmg_pipeline.py:37-140` |
| 运动生成：`gen_motion_sequence()` 滑动窗口，stride=n_motions=100 帧（4s @25fps），末窗 padding + indicator 掩码 | `live_portrait_wmg_wrapper.py:359-413` |
| 窗口衔接：每窗结束取 `prev_motion_feat = motion_feat[:, -n_prev_motions:]`（**n_prev=25**，checkpoint 配置覆盖了代码默认值 10） | `live_portrait_wmg_wrapper.py:412-413` |
| 采样：**完整 DDPM 反向 500 步**（`n_diff_steps=500`，cosine schedule），CFG `cfg_scale=1.15`（cfg_mode=incremental，等效 batch×2） | `dit_talking_head.py:60-75, 277-290` |
| DiT 参数：`motion_feat_dim=256`（模型配置覆盖默认 76）、`n_motions=100`、`audio_model="hubert"`、`feature_dim=512` | `src/config/models.yaml` |
| 音频入口：`sample(audio_or_feat, ...)` 支持传**原始 16k 音频**（ndim=2，64000 样本/窗）内部自行提特征，或直接传 (N,100,512) 特征（ndim=3） | `dit_talking_head.py:238-247` |
| 权重路径约定：`pretrained_weights/JoyVASA/motion_generator/motion_generator_hubert_chinese.pt`、`pretrained_weights/JoyVASA/motion_template/motion_template.pkl`、LivePortrait 五模型（F/M/W/G/S）、`insightface/models/buffalo_l`、`TencentGameMate/chinese-hubert-base` | `inference_config.py:21-23`、README §权重 |
| 依赖事实：推理链核心 import 仅 torch/numpy/cv2/librosa/yaml/rich + insightface(onnxruntime)；diffusers/xformers 仅训练用，推理代码零 import | requirements + import 扫描 |
| 当前两仓库 `pretrained_weights/` 均为空/不存在 | 实测 |

### 1.2 LiveTalking 侧（D:\LiveTalking）

| 契约点 | 位置 | 说明 |
|---|---|---|
| avatar 注册 | `app.py:131-140` `_avatar_modules` 字典 + `registry.create("avatar", opt.model, opt, model, avatar)` | 新模型需在此登记模块名，并提供模块级 `load_model()/load_avatar()/warm_up()` |
| 模型注册装饰器 | `avatars/musetalk_avatar.py:110` `@register("avatar", "musetalk")` | 模板写法 |
| 渲染主循环 | `base_avatar.py:327-382` `inference()`：`feat_queue.get()` 取 batch → `self.inference_batch(index, audiofeat_batch)` 返回 batch 帧 → 入 `res_frame_queue` | 全静音时跳过推理只取背景帧 |
| 贴回调用 | `base_avatar.py:434` `paste_back_frame(res_frame, idx)` | 每帧调用，异常被捕获跳过 |
| ASR 基类 | `avatars/audio_features/base_asr.py`：`run_step()` 由 `base_avatar.py:490` 每轮驱动；`stride_left/right = opt.l/r`；`feat_queue` maxsize=2 | 特征分块用 `_feature2chunks(feature_idx_multiplier=2)`（hubert 50Hz→25fps） |
| 现有 hubert ASR | `avatars/audio_features/hubert.py`（hubert-large-ls960-ft 英文，1024 维） | 与 JoyVASA 的 chinese-hubert-base（512 维投影后）**不同**，须新写 |
| 音频帧规格 | 16kHz float32、320 样本/chunk（20ms）、25fps | `BaseASR.__init__` |
| 形象预处理分发 | `server/task_manager.py:88-114` 按 `model_type` 调 `avatars/<model>/genavatar.py::generate_avatar(video_path, avatar_id, save_path, **专有参数, progress_callback)` | 需新增 joyvasa 分支 |
| 形象数据目录 | `data/avatars/<avatar_id>/`（musetalk: full_imgs/ + coords.pkl + latents.pt …） | joyvasa 沿用同目录布局 |
| 配置 | `config.py:61-66`：`--model`（choices 默认 wav2lip）、`--avatar_id`、`--batch_size 16`、`--fps 25` | help 文案需更新 |
| 启动器 | `tools/start_livetalking.py` 固定 `--model wav2lip` | 可选：加 `--model` 透传 |

---

## 2. 架构与数据流

```
TTS (mimotts/edgetts)                       16kHz PCM, 320样本/20ms
      │ put_audio_frame
      ▼
ChHubertASR(BaseASR)          ←新写：不做特征提取，只做缓冲/静音判定，
      │ run_step(): 每batch推 batch_size 个 PCM 帧进 feat_queue
      ▼
JoyVASAAvatar.inference()  (继承 BaseAvatar，复用主循环)
      │ inference_batch(index, audiofeat_batch)
      ▼
┌─ JoyVASAAvatar.inference_batch ──────────────────────────────┐
│ 1. audiofeat_batch 追加进 rolling PCM 缓冲                    │
│ 2. 缓冲 ≥ 100 帧(64000样本) → motion_generator.sample(       │
│      audio_window, prev_motion_feat, prev_audio_feat)        │
│    → 100 帧运动系数(exp/scale/t/R) 入 deque，更新 prev 链      │
│ 3. 从 deque 弹 batch_size(16) 帧 → 逐帧构造 x_d_i_new         │
│    (R/delta/scale/t 相对运动公式，移植 pipeline:100-130)       │
│ 4. warp_decode(f_s[idx], x_s[idx], x_d_i_new) 批量渲染        │
│    → 返回 batch_size 张 512×512 crop 帧                       │
└──────────────────────────────────────────────────────────────┘
      ▼ res_frame_queue
process_frames → paste_back_frame(res_frame, idx)
      │ 用预存的 M_c2o + mask_ori 贴回 full_imgs[idx] 原帧
      ▼
WebRTC 推流（output.push_video_frame，25fps）
```

要点：
- **音频特征不用 LiveTalking 的 hubert 通路**。DiT 需要「整窗 4 秒音频」而非「逐帧 ±8 帧特征窗」，强行对齐会破坏模型时序建模。因此 ASR 层退化为 PCM 缓冲器，`sample()` 传原始音频（ndim=2 路径），特征提取完全复用 JoyVASA 内部实现——**零适配风险**。
- **首帧延迟发生在第 1 窗**：缓冲攒满 4 秒语音才触发第一次采样；此后采样与播放流水线化（播第 N 窗时采第 N+1 窗）。
- **静音**：base_avatar 主循环已自动跳过（走背景帧），缓冲在静音期保留但不推进采样。

---

## 3. 文件级改造清单

### 3.1 新增文件

| 文件 | 内容 | 来源 |
|---|---|---|
| `avatars/joyvasa/`（目录） | `modules/`：appearance_feature_extractor / motion_extractor / warping_network / spade_generator / stitching_retargeting_network / dense_motion / convnextv2 / common / util + `dit_talking_head.py` + `hubert.py`（chinese 版） + `wav2vec2.py` | 从 `D:\JoyVASA\src\modules\` **原样复制**（去掉 __init__ 中的训练项） |
| `avatars/joyvasa/utils/` | `crop.py`（prepare_paste_back/paste_back）、`cropper.py`、`face_analysis_diy.py`、`human_landmark_runner.py`、`helper.py`、`io.py`、`rprint.py`、`timer.py`、`camera.py`(可裁) | 从 `D:\JoyVASA\src\utils\` 复制并裁剪 |
| `avatars/joyvasa/config.py` | 精简版 InferenceConfig/CropConfig/models.yaml，**路径改为 `./models/joyvasa/...`**（相对 LiveTalking 根目录） | 改写自 `src/config/` |
| `avatars/joyvasa/wmg_wrapper.py` | LivePortraitWrapper + gen_motion_sequence 核心（裁掉 gradio/template 动画分支），暴露 `sample_motion(audio_window, prev_state) -> 100 coefs` | 改写自 `live_portrait_wmg_wrapper.py` |
| `avatars/joyvasa/genavatar.py` | `generate_avatar(video_path, avatar_id, save_path, ...)`：逐帧 insightface 检测→crop 256→M_c2o/mask/lmk→计算并缓存 I_s/f_s/x_s_info/x_s/R_s→存 `data/avatars/<id>/`（full_imgs/ + source_cache.pt fp16 + meta.json） | 新写，参照 `musetalk/genavatar.py` 结构 |
| `avatars/joyvasa_avatar.py` | `@register("avatar","joyvasa") class JoyVASAReal(BaseAvatar)` + 模块级 `load_model()/load_avatar()/warm_up()` | 新写（musetalk_avatar.py 为模板） |
| `avatars/audio_features/pcmbuf.py` | `PcmBufferASR(BaseASR)`：run_step 推 PCM 帧进 feat_queue；`flush_talk` 清空缓冲与 prev 链 | 新写（仿 hubert.py 骨架，去掉特征提取） |

### 3.2 修改文件（每处 ≤5 行）

| 文件 | 改动 |
|---|---|
| `app.py:131-135` | `_avatar_modules` 增加 `'joyvasa': 'avatars.joyvasa_avatar'` |
| `config.py:61-62` | `--model` help 改为 `musetalk/wav2lip/ultralight/joyvasa`（argparse 无 choices 强约束，无需其他改动） |
| `server/task_manager.py:114` | elif 分支：`joyvasa` → `avatars.joyvasa.genavatar.generate_avatar(...)`（专有参数：`det_size`、`persistence`） |
| `tools/start_livetalking.py` | （可选）增加 `--model` 透传与 `--mode joyvasa` 组合 |
| `docs/full-local-deployment-guide.md` | 新增第 4 章：joyvasa 形象制作 + 启动命令 + 权重清单 |
| `.gitignore` | 已有 `models/` 忽略，无需改；确认 `data/` 已忽略 |

### 3.3 不动的部分

TTS 三层（mimotts 等）、streamout、registry 机制、BaseAvatar 主循环——全部零改动，这正是该插件架构的收益点。

---

## 4. 分阶段实施步骤

### Phase 0：风险验证（半天，决定整个方案是否成立）✅ 先做

**0.1 依赖安装**（venv 内，不动 torch）：
```bash
cd D:\LiveTalking
.venv\Scripts\pip install insightface==0.7.3 onnxruntime-gpu==1.18.0 onnx==1.16.1
```
> insightface 需 C++ Build Tools 编译或有预编译 wheel（Windows + py3.12 有社区 wheel；若失败用 `pip install insightface --only-binary :all:` 或降级方案见 §6）

**0.2 下载 chinese-hubert-base + JoyVASA motion generator**（约 700MB，走 hf-mirror.com）：
```bash
set HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download TencentGameMate/chinese-hubert-base --local-dir models/joyvasa/chinese-hubert-base
huggingface-cli download jdh-algo/JoyVASA --local-dir models/joyvasa/JoyVASA
```

**0.3 三项实测**（写 `tools/test_joyvasa_risk.py`）：
1. transformers 5.19 加载 JoyVASA 版 `HubertModel`（override 了 forward，用 `feature_extractor/feature_projection` 内部 API）→ 768 维输出正常；
2. `DitTalkingHead` 加载 `motion_generator_hubert_chinese.pt`，喂 64000 样本随机音频跑一次 `sample()` → 记录耗时（这是首帧延迟的实测值）、输出形状 (1,100,256)；
3. insightface buffalo_l 对测试图检测+crop 正常。

**验收**：三项全过 → 继续 Phase 1；任一失败 → 走 §6 备选（transformers 降级 4.39 / 采样步数缩减）后重测。

### Phase 1：权重下载（0.5 天，可与 Phase 0 并行）

```bash
huggingface-cli download KwaiVGI/LivePortrait --local-dir models/joyvasa/liveportrait --exclude "*.git*" "README.md" "docs"
# insightface 人脸分析模型
huggingface-cli download public-data/buffalo_l --local-dir models/joyvasa/insightface/models/buffalo_l
```
最终目录（约 3.5GB）：
```
models/joyvasa/
├── liveportrait/            # F/M/W/G/S 五个 .pth
├── JoyVASA/
│   ├── motion_generator/motion_generator_hubert_chinese.pt
│   └── motion_template/motion_template.pkl
├── chinese-hubert-base/
└── insightface/models/buffalo_l/
```
**验收**：`load_model` 五模型 + DiT 全部加载到显存，总计 <6GB（2080Ti 22GB 富余）。

### Phase 2：代码移植（3-4 天）

按 §3.1 清单执行，顺序：
1. **Day 1**：复制裁剪 `avatars/joyvasa/`（modules + utils + config），跑通「单图 → sample() 运动 → warp_decode 逐帧渲染 → paste_back」离线管线（一个 `tools/test_joyvasa_offline.py` 脚本，输入 wav + 参考图，输出 mp4）——**这是最重要的一天**，全部数学在脱离 LiveTalking 的情况下验证；
2. **Day 2**：写 `pcmbuf.py` + `joyvasa_avatar.py`，实现滑动窗口缓冲、prev 链管理、flush_talk 清理；注册进 app.py/config.py，`--model joyvasa` 能起服务；
3. **Day 3**：写 `genavatar.py` + task_manager 分支，用一段真人视频制作 avatar，验证 full_imgs/source_cache 加载；
4. **Day 4**：端到端联调：网页 → TTS → 数字人说话，修同步/状态切换 bug。

### Phase 3：联调与优化（1-2 天）

- **延迟优化**（按收益顺序试）：
  1. `cfg_cond=[]` 关闭 CFG → 采样 batch 减半（画质微降，延迟约减半）；
  2. DDIM 子采样：500 步 → 50 步（改 `sample()` 的 t 序列，需小改 dit_talking_head.py，实测画质损失）；
  3. fp16：五模型 + DiT 全 `.half()`（LivePortrait 官方支持）；
  4. torch.compile（LivePortrait 原生支持，再提速 20-30%）。
- **同步校验**：log 里 `actual avg infer fps` 应 ≥ 25；观察音频/口型偏移，必要时调 batch_size。
- **静音→说话切换**：确认缓冲重置不产生跳帧（复用 base_avatar 的 mirror_index 机制）。

**验收**：一句话回复（~10 秒音频）端到端：首帧 ≤10s，之后连续流畅；长对话 30 分钟无显存泄漏。

### Phase 4：固化（0.5 天）

- 更新 `docs/full-local-deployment-guide.md`（第 4 章）与 README 模型列表；
- 一键启动器支持 `start.bat --model joyvasa`；
- **提交并推送到 fork**（本次已建立的标准流程），建议 2 个提交：`feat: joyvasa avatar module` + `docs: joyvasa integration guide`。

---

## 5. 实时性预算（2080Ti 22GB 估算，Phase 0 实测修正）

| 环节 | 估算 | 说明 |
|---|---|---|
| DiT 采样（500 步 DDPM，CFG×2） | **3-10s** | 首帧延迟主体；优化后可压到 1-3s |
| warp_decode 单帧渲染 | 10-20ms | 批量 16 帧更快 |
| paste_back 单帧 | <2ms | 纯 numpy/cv2 |
| hubert 特征提取（DiT 内部） | ~100ms/窗 | 与采样流水线化，非瓶颈 |
| 稳态吞吐 | 播 4s 音频的时间内完成下窗采样 | 只要采样 ≤4s 即无缝衔接 |

---

## 6. 风险与备选

| 风险 | 概率 | 备选方案 |
|---|---|---|
| transformers 5.19 不兼容 JoyVASA hubert 内部 API | 中 | ① 该模块单独用 `pip install "transformers==4.39.2"` 装到独立目录 + sys.path 优先加载（不污染 venv）；② 改走 wav2vec2-base 音频编码器（JoyVASA 官方同样支持，facebook/wav2vec2-base-960h 走标准 API 兼容性更好） |
| insightface Windows py3.12 编译失败 | 中 | 用预编译 wheel（GitHub community builds）；或 genavatar 阶段用 LiveTalking 现成的 s3fd 检测 + insightface 仅做 lmk（onnxruntime CPU 版即可，检测在预处理阶段非实时） |
| DDPM 500 步实测 >10s | 低-中 | DDIM 50 步 + 关 CFG，论文级画质换实时性；仍不达标则此模型定位为「预渲染模式」（离线生成视频再循环播放，类似 customvideo） |
| 显存超限 | 极低 | 22GB 上限，五模型+DiT fp16 约 6GB；与 LLM 无同卡冲突 |
| 采样窗口与 TTS 打断冲突 | 低 | flush_talk 时清空 PCM 缓冲 + prev 链，丢弃未采样窗口；已采样未播放的 motion 出队丢弃（与 musetalk flush 行为一致） |

**回滚**：所有改动均为新增文件 + 5 处 ≤5 行修改，`git revert` 两个提交即完全回滚，不影响现有 wav2lip/musetalk/ultralight 通路。

---

## 7. 附录：启动命令（Phase 4 完成后）

```bash
# 制作形象（API 或脚本）
# server API: POST /record 下发 model_type=joyvasa 的 genavatar 任务

# 启动服务
.venv\Scripts\python.exe app.py --transport webrtc --model joyvasa \
    --avatar_id my_joyvasa_avatar --listenport 8010 --tts mimotts \
    --REF_FILE 冰糖 --MIMO_STYLE "自然温柔的女声，语速适中"

# 一键
start.bat --model joyvasa
```
