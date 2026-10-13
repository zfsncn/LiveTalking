# -*- coding: utf-8 -*-
"""LiveTalking 一键启动器
=====================

双击项目根目录的 start.bat 即可：
  1. 自动清理占用 8010 的旧进程（避免 WinError 10048 端口冲突）
  2. 以 mimotts 模式（默认）启动 LiveTalking
  3. 等待服务就绪后自动打开浏览器 http://localhost:8010/index.html
  4. 运行日志同步写入 logs/start-<时间戳>.log

用法（一般不用手动敲，双击 start.bat 即可）：
  start.bat                      # 默认：模式 C（MiMo-V2.5-TTS，音色「冰糖」）
  start.bat --mode b             # 模式 B：edge-tts 免费云端语音
  start.bat --avatar wav2lip_avatar_long_hair_girl   # 换形象
  start.bat --port 8011 --no-browser                 # 换端口 / 不弹浏览器
"""

import argparse
import datetime as _dt
import os
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NO_WINDOW = subprocess.CREATE_NO_WINDOW  # 后台命令不弹黑窗

MODES = {
    "c": {
        "desc": "MiMo-V2.5-TTS 云端音色（默认，无需本地 TTS 服务）",
        "args": ["--tts", "mimotts", "--REF_FILE", "冰糖",
                 "--MIMO_STYLE", "自然温柔的女声，语速适中"],
    },
    "b": {
        "desc": "edge-tts 免费云端语音（需已打 403 补丁）",
        "args": ["--tts", "edgetts"],
    },
}

BASE_ARGS = ["--transport", "webrtc", "--model", "wav2lip"]


def log(msg=""):
    print(msg, flush=True)


def free_port(port):
    """结束占用 port 的进程，返回被杀掉的 PID 列表（避免 WinError 10048）。"""
    try:
        out = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True,
            errors="replace", creationflags=NO_WINDOW,
        ).stdout
    except Exception:
        return []
    pids = []
    for line in out.splitlines():
        parts = line.split()
        if (len(parts) >= 4 and parts[0] in ("TCP", "TCP6")
                and parts[3] == "LISTENING"
                and parts[1].endswith(":%d" % port)):
            pid = parts[-1]
            if pid.isdigit() and int(pid) != os.getpid() and pid not in pids:
                pids.append(pid)
    for pid in pids:
        subprocess.run(["taskkill", "/PID", pid, "/F", "/T"],
                       capture_output=True, creationflags=NO_WINDOW)
    return pids


def stream_output(proc, logfile):
    """把子进程输出同时写到控制台和日志文件。"""
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        logfile.write(line)
        logfile.flush()
    try:
        proc.stdout.close()
    except Exception:
        pass


def wait_healthy(port, proc, timeout):
    """轮询等待 HTTP 200；子进程提前退出则立即返回 False。"""
    url = "http://127.0.0.1:%d/index.html" % port
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def check_env(mode):
    """启动前的轻量预检，返回警告列表（不阻断启动）。"""
    warnings = []
    env_file = ROOT / ".env"
    if not env_file.exists():
        warnings.append("未找到 .env（LLM / MiMo key 将不可用），"
                        "可参考 docs\\full-local-deployment-guide.md 3.6 节创建")
        return warnings
    content = env_file.read_text(encoding="utf-8", errors="replace")
    if mode == "c":
        for line in content.splitlines():
            if line.startswith("MIMO_API_KEY="):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                if not val or set(val) == {"x"}:
                    warnings.append(".env 中 MIMO_API_KEY 为空或占位符，"
                                    "mimotts 合成会失败，请填入真实 key")
                break
        else:
            warnings.append(".env 中未配置 MIMO_API_KEY，mimotts 合成会失败")
    return warnings


def main():
    parser = argparse.ArgumentParser(description="LiveTalking 一键启动器")
    parser.add_argument("--mode", choices=["b", "c"], default="c",
                        help="TTS 模式：c=MiMo(默认) / b=edge-tts")
    parser.add_argument("--port", type=int, default=8010, help="服务端口（默认 8010）")
    parser.add_argument("--avatar", default="wav2lip256_avatar1",
                        help="形象目录名，位于 data/avatars/ 下（默认 wav2lip256_avatar1）")
    parser.add_argument("--no-browser", action="store_true",
                        help="服务就绪后不自动打开浏览器")
    args = parser.parse_args()

    if ".venv" not in sys.executable:
        log("[warn] 当前解释器不是 .venv 的 Python，建议通过 start.bat 启动")

    for w in check_env(args.mode):
        log("[warn] " + w)

    # app.py 不会自建 data/record（/record 下载端点依赖它），兜底创建
    (ROOT / "data" / "record").mkdir(parents=True, exist_ok=True)

    killed = free_port(args.port)
    if killed:
        log("[info] 已停止占用端口 %d 的旧进程：PID %s"
            % (args.port, ", ".join(killed)))
        time.sleep(1)  # 等端口真正释放

    cmd = [sys.executable, str(ROOT / "app.py")] + BASE_ARGS + [
        "--avatar_id", args.avatar,
        "--listenport", str(args.port),
    ] + MODES[args.mode]["args"]

    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = ROOT / "logs"
    logdir.mkdir(exist_ok=True)
    logpath = logdir / ("start-%s.log" % ts)

    log("=" * 62)
    log(" LiveTalking 一键启动 - 模式 %s：%s"
        % (args.mode.upper(), MODES[args.mode]["desc"]))
    log(" 命令：%s" % " ".join(cmd))
    log(" 地址：http://localhost:%d/index.html" % args.port)
    log(" 日志：%s" % logpath)
    log("=" * 62)

    with open(logpath, "w", encoding="utf-8", errors="replace") as logfile:
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        threading.Thread(target=stream_output, args=(proc, logfile),
                         daemon=True).start()

        log("[info] 正在加载模型并启动服务（通常 15-40 秒）...")
        healthy = wait_healthy(args.port, proc, timeout=180)

        if not healthy:
            log("")
            log("[error] 服务启动失败或超时，请查看上方日志；日志文件：%s" % logpath)
            try:
                proc.terminate()
            except Exception:
                pass
            return 1

        log("[ok] 服务就绪：http://localhost:%d/index.html" % args.port)
        if not args.no_browser:
            try:
                webbrowser.open("http://localhost:%d/index.html" % args.port)
                log("[ok] 已调用默认浏览器打开页面")
            except Exception:
                log("[warn] 自动打开浏览器失败，请手动访问上述地址")
        log("[info] 关闭本窗口或按 Ctrl+C 可停止服务")

        try:
            proc.wait()
        except KeyboardInterrupt:
            log("")
            log("[info] 正在停止服务...")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
        return proc.returncode if proc.returncode is not None else 0


if __name__ == "__main__":
    sys.exit(main())
