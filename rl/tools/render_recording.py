"""Headless, deterministic presentation of recorded native game states."""
import argparse
from functools import lru_cache
import gzip
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from generals.core.observation import visibility_mask


SIZE = (1140, 816)
CELL = 54
BLUE, RED, GOLD = "#327fba", "#cf5268", "#ffe487"
BACKGROUND, WHITE, MUTED = "#0e1521", "#eff5ff", "#9baec5"


@lru_cache(maxsize=32)
def font(size, *, numeric=False, bold=False):
    if numeric:
        path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        return ImageFont.truetype(path, size)
    suffix = "Bold" if bold else "Regular"
    path = Path(f"/usr/share/fonts/opentype/noto/NotoSansCJK-{suffix}.ttc")
    if path.exists():
        return ImageFont.truetype(str(path), size, index=2)
    return ImageFont.truetype("DejaVuSans.ttf", size)


def _text(draw, position, text, size=20, color=WHITE, **options):
    draw.text(position, text, fill=color, font=font(size, **options))


def _army(count):
    if count < 10000:
        return str(count)
    return f"{count / 1000:.0f}k" if count < 1000000 else f"{count / 1000000:.1f}m"


def _arrow(draw, origin, source, direction, width, *, color):
    row, col = divmod(source, width)
    dy, dx = ((-1, 0), (0, 1), (1, 0), (0, -1))[direction]
    if not 0 <= row + dy < 9 or not 0 <= col + dx < 9:
        return
    x, y = origin
    start = (x + (col + .5) * CELL, y + (row + .5) * CELL)
    end = (start[0] + dx * CELL, start[1] + dy * CELL)
    draw.line((start, end), fill="#152331", width=7)
    draw.line((start, end), fill=color, width=4)
    angle = math.atan2(dy, dx)
    wing = 12
    points = [end, (end[0] - wing * math.cos(angle - .5), end[1] - wing * math.sin(angle - .5)),
              (end[0] - wing * math.cos(angle + .5), end[1] - wing * math.sin(angle + .5))]
    draw.polygon(points, fill=color)


def _board(draw, state, learner, origin, *, local, actions):
    terrain, owner, army, structure = (np.asarray(state[key]) for key in ("terrain", "owner", "army", "structure"))
    if owner.shape != (9, 9):
        raise ValueError("this presentation layout currently expects the 9x9 baseline")
    visible = visibility_mask(owner, learner) if local else np.ones_like(owner, dtype=bool)
    left, top = origin
    for row in range(9):
        for col in range(9):
            x, y = left + col * CELL, top + row * CELL
            seen, pid, building = bool(visible[row, col]), int(owner[row, col]), int(structure[row, col])
            if not seen:
                color = "#182333"
            elif terrain[row, col] == 1:
                color = "#344556"
            elif pid == learner:
                color = BLUE
            elif pid >= 0:
                color = RED
            elif building:
                color = "#62727f"
            else:
                color = "#cbd7df"
            draw.rectangle((x, y, x + CELL, y + CELL), fill=color, outline="#142030", width=1)
            if not seen:
                continue
            if terrain[row, col] == 1:
                draw.polygon([(x + 11, y + 40), (x + 27, y + 12), (x + 44, y + 40)],
                             fill="#8393a4", outline="#abb8c6")
                continue
            if building == 2:
                points = (.20, .18, .36, .31, .50, .12, .64, .31, .80, .18, .72, .44, .28, .44)
                draw.polygon([(x + points[i] * CELL, y + points[i + 1] * CELL)
                              for i in range(0, len(points), 2)], fill=GOLD)
            elif building == 1:
                draw.rectangle((x + 14, y + 13, x + 40, y + 24), fill="#f2f5f7")
                for offset in (14, 24, 34):
                    draw.rectangle((x + offset, y + 8, x + offset + 6, y + 15), fill="#f2f5f7")
            count = int(army[row, col])
            if pid >= 0 or count > 0 or building:
                number = _army(count)
                target = (x + CELL / 2, y + (CELL * .71 if building else CELL * .52))
                draw.text(target, number, anchor="mm", font=font(20, numeric=True),
                          fill=WHITE if pid >= 0 or building else "#253545",
                          stroke_width=1, stroke_fill="#29394a" if pid >= 0 or building else color)
    if actions:
        for pid, action in enumerate(actions):
            kind, source, direction, _ = action
            if kind != 1 or not 0 <= source < 81 or (local and pid != learner):
                continue
            _arrow(draw, origin, source, direction, 9, color=GOLD if pid == learner else "#ff947b")


def _action_text(frame, pid):
    if not frame["actions"]:
        return "初始局面"
    kind, source, direction, mode = frame["actions"][pid]
    if kind == 0:
        return "等待"
    row, col = divmod(source, 9)
    dy, dx = ((-1, 0), (0, 1), (1, 0), (0, -1))[direction]
    move = frame["moves"][pid]
    route = f"({row + 1},{col + 1}) → ({row + dy + 1},{col + dx + 1})"
    mode_text = "半军" if mode else "留1出兵"
    if not move[0]:
        effect = "本回合未执行"
    elif not move[1]:
        effect = "动作被规则拒绝"
    else:
        effect = f"实际移动 {move[4]}"
        if move[5]:
            effect += "，占领"
        if move[6] >= 0:
            effect += "，夺将"
    return f"{route}，{mode_text}，{effect}"


def draw_frame(recording, frame, *, ticks_per_second=8):
    metadata = recording["metadata"]
    game = metadata["game"]
    state = frame["state"]
    learner, opponent = game["learner_seat"], game["opponent"]
    tick, final_tick = state["tick"], recording["frames"][-1]["state"]["tick"]
    image = Image.new("RGB", SIZE, BACKGROUND)
    draw = ImageDraw.Draw(image)
    _text(draw, (28, 18), f"Generals · RL vs {opponent.upper()}", 28, bold=True)
    _text(draw, (28, 60), f"snapshot {metadata.get('snapshot_id', '?')}  ·  "
          f"tick {tick} / {final_tick}  ·  播放 {ticks_per_second} tick/s", 18, MUTED)
    owner, army = np.asarray(state["owner"]), np.asarray(state["army"])
    for pid, x, color, name in ((learner, 28, BLUE, f"RL 学习者 · P{learner}"),
                                (1 - learner, 600, RED, f"{opponent} · P{1 - learner}")):
        mask = owner == pid
        _text(draw, (x, 105), name, 21, color, bold=True)
        _text(draw, (x, 136), f"兵力 {int(army[mask].sum())}     领土 {int(mask.sum())}", 19)
    _text(draw, (28, 174), "全图观看（不传给策略）", 20, MUTED)
    _text(draw, (600, 174), "RL 当前局部视野（另用4帧历史）", 20, MUTED)
    _board(draw, state, learner, (28, 207), local=False, actions=frame["actions"])
    _board(draw, state, learner, (600, 207), local=True, actions=frame["actions"])
    _text(draw, (28, 707), f"RL 上一步：{_action_text(frame, learner)}", 18, GOLD)
    _text(draw, (28, 739), f"对手上一步：{_action_text(frame, 1 - learner)}", 18, "#ffae9c")
    _text(draw, (28, 781), "蓝色=RL  红色=baseline  王冠=将军  城墙=城市  三角=山  箭头=上一步动作", 15, MUTED)
    if state["terminated"]:
        won = state["winner_id"] == learner
        draw.rounded_rectangle((907, 19, 1112, 78), radius=14,
                               fill="#237652" if won else "#92384c")
        _text(draw, (935, 29), "RL 获胜" if won else "RL 落败", 28, bold=True)
    return image


def render_recording(frames_path, output, *, ticks_per_second=8, fps=24):
    if ticks_per_second < 1 or fps < 1 or fps % ticks_per_second:
        raise ValueError("fps must be a positive multiple of ticks_per_second")
    with gzip.open(frames_path, "rt", encoding="utf-8") as stream:
        recording = json.load(stream)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    log = output.with_suffix(".ffmpeg.log")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
               "-pixel_format", "rgb24", "-video_size", f"{SIZE[0]}x{SIZE[1]}",
               "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
               "-preset", "veryfast", "-crf", "20", "-threads", "2",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    frame_count = 0
    with log.open("w") as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
        try:
            for index, frame in enumerate(recording["frames"]):
                rendered = draw_frame(recording, frame, ticks_per_second=ticks_per_second)
                if index in (0, len(recording["frames"]) // 2, len(recording["frames"]) - 1):
                    suffix = "start" if index == 0 else ("end" if index == len(recording["frames"]) - 1 else "middle")
                    rendered.save(output.with_name(f"{output.stem}_{suffix}.png"))
                repetitions = fps * 2 if index == 0 else fps // ticks_per_second
                if index == len(recording["frames"]) - 1:
                    repetitions += fps * 4
                data = rendered.tobytes()
                for _ in range(repetitions):
                    process.stdin.write(data)
                    frame_count += 1
            process.stdin.close()
            returncode = process.wait()
            if returncode:
                raise RuntimeError(f"ffmpeg failed with code {returncode}; see {log}")
        except BaseException:
            if not process.stdin.closed:
                process.stdin.close()
            if process.poll() is None:
                process.kill()
            process.wait()
            raise
    return {"video": str(output.resolve()), "frames": frame_count, "fps": fps,
            "duration_seconds": frame_count / fps, "width": SIZE[0], "height": SIZE[1],
            "ticks_per_second": ticks_per_second, "bytes": output.stat().st_size}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(render_recording(args.frames, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
