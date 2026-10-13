"""
MiMo-V2.5-TTS via OpenAI-compatible chat completions audio API.

POST {base_url}/chat/completions  (OpenAI SDK, audio 参数)
  messages = [
      {"role": "user", "content": "<风格指令>"},
      {"role": "assistant", "content": "<待合成文本>"},
  ]
  audio = {"format": "wav", "voice": "<音色名>"}
返回 completion.choices[0].message.audio.data = base64 WAV。

模型事实（官方文档 + 实测）：
  - 输出采样率固定 24000 Hz → 需重采样到项目标准 16000 Hz
  - 预置音色 8 个：冰糖(默认)/茉莉/苏打/白桦（中文）+ Mia/Chloe/Milo/Dean（英文）
  - 支持自然语言风格指令（user 消息）与文本内嵌音频标签，如 (开心)(温柔)(唱歌)
  - 公测期间免费；限流 RPM 100 / TPM 10M
  - 也支持流式（pcm16），本实现为非流式 v1（整句返回后分块推送）

需要环境变量 MIMO_API_KEY（写入 .env，dotenv 自动加载）。

用法:
    python app.py --tts mimotts --REF_FILE 冰糖 --MIMO_STYLE "自然温柔的女声，语速适中"
"""

import os
import io
import time
import base64
from typing import Optional

import numpy as np
import resampy
import soundfile as sf
from openai import OpenAI

from utils.logger import logger
from .base_tts import BaseTTS, State
from registry import register

# MiMo 官方固定输出采样率（WAV 头里读到的真实值优先，此为兜底）
DEFAULT_SRC_SR = 24000

# 默认风格指令（user 消息）：中文数字人的稳妥默认
DEFAULT_STYLE = "自然温柔的女声，语速适中，吐字清晰，情绪平稳"

DEFAULT_BASE_URL = "https://api.xiaomimimo.com/v1"
DEFAULT_MODEL = "mimo-v2.5-tts"
DEFAULT_VOICE = "冰糖"  # 中文女声（官方默认音色）


@register("tts", "mimotts")
class MiMoTTS(BaseTTS):
    """TTS client for Xiaomi MiMo-V2.5-TTS (OpenAI-compatible audio chat API)."""

    def __init__(self, opt, parent):
        super().__init__(opt, parent)

        # ── 凭据与端点 ─────────────────────────────────────────
        self.api_key = os.environ.get("MIMO_API_KEY", "") or ""
        self.base_url = getattr(opt, "MIMO_BASE_URL", None) or DEFAULT_BASE_URL
        self.model = getattr(opt, "MIMO_MODEL", None) or DEFAULT_MODEL

        # ── 音色与风格 ─────────────────────────────────────────
        # REF_FILE 复用为音色名（与 omnitts/qwentts 惯例一致）
        self.voice = opt.REF_FILE or DEFAULT_VOICE
        # MIMO_STYLE 为默认风格指令；per-message 可用 datainfo.tts.instructions 覆盖
        self.style = getattr(opt, "MIMO_STYLE", None) or DEFAULT_STYLE
        self.audio_format = getattr(opt, "MIMO_FORMAT", None) or "wav"

        self.src_sr = int(getattr(opt, "MIMO_SRC_SR", DEFAULT_SRC_SR))
        self.timeout = float(getattr(opt, "MIMO_TIMEOUT", 60))

        if not self.api_key:
            logger.error(
                "MiMoTTS: MIMO_API_KEY 未设置（请在 .env 中配置），"
                "合成调用将失败"
            )

        self.client = OpenAI(
            api_key=self.api_key or "not-needed",
            base_url=self.base_url,
            timeout=self.timeout,
        )

        logger.info(
            f"MiMoTTS init: base_url={self.base_url}, model={self.model}, "
            f"voice={self.voice}, style={self.style[:40]}..."
        )

    # ── main entry ────────────────────────────────────────────

    def txt_to_audio(self, msg: tuple[str, dict]):
        text, textevent = msg

        # per-message overrides（与 omnitts 同机制）
        tts_cfg = textevent.get("tts", {})
        voice = tts_cfg.get("ref_file", self.voice)
        style = tts_cfg.get("instructions", self.style)

        audio_bytes = self._synthesize(text=text, voice=voice, style=style)
        if not audio_bytes:
            logger.error("MiMoTTS: 合成失败，无音频返回")
            return

        self._decode_and_push(audio_bytes, text, textevent)

    # ── API call（非流式，整句返回） ────────────────────────────

    def _synthesize(self, text: str, voice: str, style: str) -> Optional[bytes]:
        """
        调用 MiMo chat completions audio API，返回解码后的 WAV 字节。
        """
        start = time.perf_counter()
        logger.info(f"MiMoTTS POST {self.base_url}/chat/completions "
                    f"voice={voice} text={text[:60]}...")

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": style},
                    {"role": "assistant", "content": text},
                ],
                audio={
                    "format": self.audio_format,
                    "voice": voice,
                },
            )

            message = completion.choices[0].message
            audio = getattr(message, "audio", None)
            b64_data = getattr(audio, "data", None) if audio else None

            if not b64_data:
                logger.error(f"MiMoTTS: 响应无 audio.data，message={message}")
                return None

            latency = time.perf_counter() - start
            logger.info(f"MiMoTTS synthesize latency: {latency:.2f}s")

            return base64.b64decode(b64_data)

        except Exception:
            logger.exception("MiMoTTS synthesize error")
            return None

    # ── 音频解码 → 重采样 → 20ms 分块推送 ──────────────────────
    # 逻辑与 omnitts.stream_tts 一致：WAV 用 soundfile 解码，
    # 非 16kHz 则重采样，按 self.chunk（320 样本 = 20ms @16kHz）推送。

    def _decode_and_push(self, wav_bytes: bytes, text: str, textevent: dict):
        try:
            with io.BytesIO(wav_bytes) as buf:
                stream, wav_sr = sf.read(buf, dtype="float32", always_2d=False)
        except Exception:
            logger.exception("MiMoTTS: WAV 解码失败")
            return

        if stream.ndim > 1:
            logger.info(f"[WARN] audio has {stream.shape[1]} channels, only use the first.")
            stream = stream[:, 0]

        logger.info(f"[INFO] tts audio stream {wav_sr}: {stream.shape}")

        if wav_sr != self.sample_rate and stream.shape[0] > 0:
            logger.info(f"[WARN] audio sample rate is {wav_sr}, resampling into {self.sample_rate}.")
            stream = resampy.resample(
                x=stream, sr_orig=wav_sr, sr_new=self.sample_rate
            )

        streamlen = stream.shape[0]
        idx = 0
        while streamlen >= self.chunk and self.state == State.RUNNING:
            eventpoint = {}
            streamlen -= self.chunk
            if idx == 0:
                eventpoint = {"status": "start", "text": text}
            elif streamlen < self.chunk:
                eventpoint = {"status": "end", "text": text}
            eventpoint.update(**textevent)
            self.parent.put_audio_frame(stream[idx:idx + self.chunk], eventpoint)
            idx += self.chunk

        # 与 omnitts 一致：补一个 end 标记的零块，确保 avatar 侧收到结束事件
        eventpoint = {"status": "end", "text": text}
        eventpoint.update(**textevent)
        self.parent.put_audio_frame(
            np.zeros(self.chunk, dtype=np.float32), eventpoint
        )

    def stop_tts(self):
        logger.info("MiMoTTS stop")
