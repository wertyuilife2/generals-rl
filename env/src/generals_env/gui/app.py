"""Single-threaded Tk application over the environment's public interfaces."""

from __future__ import annotations

import secrets
import time
import tkinter as tk
from tkinter import font as tkfont, messagebox, ttk

from ..config import MapConfig
from ..controllers import make_controllers
from ..mapgen import generate_map
from ..observation import snapshot
from ..runner import Runner
from .board import Board, PLAYER_COLORS
from .input import InputBindings

AI_KINDS = ("aggressive", "expansion", "defensive", "random")
BG = "#101821"
PANEL = "#1b2735"
TEXT = "#e9f0f6"
MUTED = "#9cabbc"


class RealtimeClock:
    """Accumulates complete ticks without dropping debt when the UI is busy."""

    def __init__(self):
        self.last = time.monotonic()
        self.debt = 0.0

    def reset(self, now=None, *, clear_debt=True):
        self.last = time.monotonic() if now is None else now
        if clear_debt:
            self.debt = 0.0

    def accumulate(self, speed, now=None):
        now = time.monotonic() if now is None else now
        self.debt += max(0.0, now - self.last) * 2.0 * speed
        self.last = now

    def consume(self):
        if self.debt < 1:
            return False
        self.debt -= 1
        return True


class App:
    def __init__(self, root, config: MapConfig, seed=42, mode="play", human_seat=0,
                 visibility="local", kinds=None, speed=1.0):
        self.root = root
        self.closed = False
        self.config = config
        self.seed = int(seed)
        self.mode = mode
        self.human_seat = human_seat
        self.visibility = visibility
        self.kinds = list(kinds) if kinds is not None else [AI_KINDS[i % 4] for i in range(config.players)]
        self.speed = float(speed)
        if self.speed <= 0:
            raise ValueError("speed must be positive")
        self.session = "PREVIEW"
        self.spectating = mode != "play"
        self.clock = RealtimeClock()
        self._callback = None
        self._view = None
        self._rate_since = time.monotonic()
        self._rate_count = 0
        self.actual_rate = 0.0
        self._notice = ""
        self._settings_window = None
        self.font_family, self.chinese = self._choose_font()
        self.root.title("Generals · CPU")
        self.root.geometry("1180x800")
        self.root.minsize(850, 650)
        self.root.configure(background=BG)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._style()
        self._build_ui()
        self.input = InputBindings(self)
        self._create_initial()
        self._load_initial()
        self._callback = self.root.after(33, self._frame)

    def _choose_font(self):
        families = set(tkfont.families(self.root))
        for family in ("Microsoft YaHei UI", "Microsoft YaHei", "Noto Sans CJK SC", "Noto Sans SC", "Source Han Sans SC", "WenQuanYi Micro Hei", "PingFang SC", "SimHei"):
            if family in families:
                return family, True
        return ("DejaVu Sans" if "DejaVu Sans" in families else "TkDefaultFont"), False

    def tr(self, chinese, english):
        return chinese if self.chinese else english

    def _style(self):
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TFrame", background=PANEL)
        style.configure("TLabel", background=PANEL, foreground=TEXT, font=(self.font_family, 10))
        style.configure("Muted.TLabel", foreground=MUTED)
        style.configure("Title.TLabel", font=(self.font_family, 18, "bold"))
        style.configure("TButton", padding=(8, 6), font=(self.font_family, 10), background="#2c4056", foreground=TEXT)
        style.map("TButton", background=[("active", "#3a5977"), ("disabled", "#243140")], foreground=[("disabled", "#637486")])
        style.configure("Accent.TButton", background="#357dab")
        style.configure("TCombobox", padding=4, font=(self.font_family, 10))
        style.configure("TCheckbutton", background=PANEL, foreground=TEXT, font=(self.font_family, 10))
        style.configure("Treeview", background=PANEL, fieldbackground=PANEL, foreground=TEXT, rowheight=25, borderwidth=0, font=(self.font_family, 10))
        style.configure("Treeview.Heading", background="#293b50", foreground=TEXT, font=(self.font_family, 10, "bold"))
        style.map("Treeview", background=[("selected", "#364d67")])

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        top = ttk.Frame(self.root, padding=(16, 10))
        top.grid(row=0, column=0, sticky="ew")
        ttk.Label(top, text="GENERALS", style="Title.TLabel").pack(side="left")
        ttk.Label(top, text=self.tr("  /  CPU 原型", "  /  CPU prototype"), style="Muted.TLabel").pack(side="left", pady=(5, 0))
        self.header_var = tk.StringVar()
        ttk.Label(top, textvariable=self.header_var, style="Muted.TLabel").pack(side="right")
        content = ttk.Frame(self.root)
        content.grid(row=1, column=0, sticky="nsew")
        content.rowconfigure(0, weight=1)
        content.columnconfigure(0, weight=1)
        self.board = Board(content, font_family=self.font_family, takefocus=True)
        self.board.grid(row=0, column=0, sticky="nsew")
        sidebar = ttk.Frame(content)
        sidebar.grid(row=0, column=1, sticky="ns")
        sidebar.rowconfigure(0, weight=1)
        self.sidebar_canvas = tk.Canvas(sidebar, width=315, background=PANEL, highlightthickness=0)
        self.sidebar_canvas.grid(row=0, column=0, sticky="ns")
        scroll = ttk.Scrollbar(sidebar, orient="vertical", command=self.sidebar_canvas.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.sidebar_canvas.configure(yscrollcommand=scroll.set)
        side = ttk.Frame(self.sidebar_canvas, padding=14)
        side_window = self.sidebar_canvas.create_window(0, 0, anchor="nw", window=side, width=315)
        side.bind("<Configure>", lambda _event: self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all")))
        self.sidebar_canvas.bind("<Configure>", lambda event: self.sidebar_canvas.itemconfigure(side_window, width=event.width))
        self.session_var = tk.StringVar()
        ttk.Label(side, textvariable=self.session_var, font=(self.font_family, 15, "bold")).pack(anchor="w", pady=(0, 4))
        self.clock_var = tk.StringVar()
        self.clock_label = ttk.Label(side, textvariable=self.clock_var, style="Muted.TLabel")
        self.clock_label.pack(anchor="w")
        self.scoreboard = ttk.Treeview(side, columns=("player", "land", "army"), show="headings", height=8, selectmode="none")
        self.scoreboard.heading("player", text=self.tr("玩家 / 策略", "Player / strategy"))
        self.scoreboard.heading("land", text=self.tr("领土", "Land"))
        self.scoreboard.heading("army", text=self.tr("兵力", "Army"))
        self.scoreboard.column("player", width=149, stretch=True)
        self.scoreboard.column("land", width=48, anchor="e", stretch=False)
        self.scoreboard.column("army", width=64, anchor="e", stretch=False)
        self.scoreboard.pack(fill="x", pady=(12, 15))
        for player, color in enumerate(PLAYER_COLORS):
            self.scoreboard.tag_configure(f"p{player}", foreground=color)
        self.scoreboard.tag_configure("dead", foreground="#647282")
        self.selection_var = tk.StringVar()
        ttk.Label(side, textvariable=self.selection_var, wraplength=280, justify="left").pack(anchor="w", pady=(0, 8))
        commands = ttk.Frame(side)
        commands.pack(fill="x")
        commands.columnconfigure((0, 1), weight=1)
        self.human_buttons = []
        entries = [
            ("选择主城 H", "General H", "home"),
            ("半兵 Z", "Half army Z", "toggle_half"),
            ("撤销末步 E", "Undo E", "undo"),
            ("清空队列 Q", "Clear queue Q", "clear_queue"),
            ("取消选择", "Deselect", "deselect"),
        ]
        for index, (cn, en, command) in enumerate(entries):
            button = ttk.Button(commands, text=self.tr(cn, en), command=lambda c=command: self.human_command(c))
            button.grid(row=index // 2, column=index % 2, sticky="ew", padx=(0, 4), pady=3)
            self.human_buttons.append(button)
            if command == "toggle_half":
                self.half_button = button
        ttk.Label(side, text=self.tr("鼠标选格 → 点击相邻格排队\n右键 / 中键拖动 · 滚轮缩放\nWASD / 方向键移动 · P 暂停", "Select tile, then adjacent tiles\nRight / middle drag; wheel zoom\nWASD / arrows move; P pauses"), style="Muted.TLabel", wraplength=285, justify="left").pack(anchor="w", pady=(12, 10))
        self.spectate_button = ttk.Button(side, text=self.tr("继续观战", "Continue spectating"), style="Accent.TButton", command=self.continue_spectating)
        self.detail_var = tk.StringVar()
        ttk.Label(side, textvariable=self.detail_var, style="Muted.TLabel", wraplength=285, justify="left").pack(anchor="w", pady=(6, 0))
        bottom = ttk.Frame(self.root, padding=(12, 10))
        bottom.grid(row=2, column=0, sticky="ew")
        self.run_button = ttk.Button(bottom, style="Accent.TButton", command=self.toggle_running)
        self.run_button.pack(side="left", padx=(0, 5))
        self.step_button = ttk.Button(bottom, text=self.tr("单步", "Step"), command=self.single_step)
        self.step_button.pack(side="left", padx=3)
        ttk.Button(bottom, text=self.tr("同图重开", "Restart"), command=self.restart).pack(side="left", padx=3)
        ttk.Button(bottom, text=self.tr("新地图", "New map"), command=self.new_map).pack(side="left", padx=3)
        self.settings_button = ttk.Button(bottom, text=self.tr("设置", "Settings"), command=self.open_settings)
        self.settings_button.pack(side="left", padx=3)
        self.speed_var = tk.StringVar(value=f"{self.speed:g}×")
        self.speed_combo = ttk.Combobox(bottom, textvariable=self.speed_var, values=("1×", "2×", "5×", "10×"), width=4, state="readonly")
        self.speed_combo.bind("<<ComboboxSelected>>", self._speed_changed)
        self.speed_combo.pack(side="left", padx=8)
        ttk.Button(bottom, text=self.tr("帮助", "Help"), command=self.show_help).pack(side="right", padx=3)
        ttk.Button(bottom, text=self.tr("适应窗口", "Fit"), command=self.board.fit).pack(side="right", padx=3)
        ttk.Button(bottom, text="+", width=2, command=lambda: self.board.zoom(1)).pack(side="right", padx=2)
        ttk.Button(bottom, text="−", width=2, command=lambda: self.board.zoom(-1)).pack(side="right", padx=2)
        self.message_var = tk.StringVar()
        ttk.Label(self.root, textvariable=self.message_var, background=BG, foreground=MUTED, padding=(16, 7), anchor="w").grid(row=3, column=0, sticky="ew")

    @property
    def human(self):
        if self.mode != "play" or self.spectating:
            return None
        return self.runner.controllers[self.human_seat]

    @property
    def human_active(self):
        return self.human is not None and bool(self.runner.engine.state.alive[self.human_seat]) and self.session != "FINISHED"

    def _create_initial(self):
        if not 2 <= self.config.players <= 8:
            raise ValueError("GUI supports 2–8 players")
        if self.mode not in {"play", "watch"}:
            raise ValueError("mode must be 'play' or 'watch'")
        if not 0 <= self.human_seat < self.config.players:
            raise ValueError("human_seat outside player range")
        if len(self.kinds) != self.config.players:
            raise ValueError("one controller kind is required per player")
        self.kinds = [AI_KINDS[i % 4] if kind == "human" else kind for i, kind in enumerate(self.kinds)]
        if self.mode == "play":
            self.kinds[self.human_seat] = "human"
        self.initial = generate_map(self.config, self.seed)

    def _load_initial(self):
        self.runner = Runner(self.initial.copy(), make_controllers(self.kinds, self.seed), seed=self.seed, visibility=self.visibility)
        self.session = "PREVIEW"
        self.spectating = self.mode != "play"
        self.clock.reset()
        self._rate_since = time.monotonic()
        self._rate_count = 0
        self.actual_rate = 0.0
        self._notice = self.tr("点击开始；暂停时也可规划行动。", "Press Start. You can plan moves while paused.")
        self.board.reset(self.config.width, self.config.height)
        self.spectate_button.pack_forget()
        self._refresh()
        self.board.focus_set()

    def restart(self):
        self._load_initial()

    def new_map(self):
        self.seed = secrets.randbits(32)
        self._create_initial()
        self._load_initial()

    def pause(self, notice=None):
        if self.session != "RUNNING":
            return
        self.session = "PAUSED"
        self.clock.reset()
        if notice:
            self._notice = notice
        self._refresh_controls()

    def toggle_running(self):
        if self.session == "RUNNING":
            self.pause()
        elif self.session != "FINISHED":
            if self.human is not None and not bool(self.runner.engine.state.alive[self.human_seat]):
                return
            self.session = "RUNNING"
            self.clock.reset()
            self._notice = ""
            self._refresh_controls()

    def single_step(self):
        if self.session in {"PREVIEW", "PAUSED"}:
            if self.human is not None and not bool(self.runner.engine.state.alive[self.human_seat]):
                return
            self.session = "PAUSED"
            self.clock.reset()
            self._notice = ""
            self._advance_tick()
            self._refresh()

    def _advance_tick(self):
        try:
            self.runner.tick()
        except Exception as error:
            # Runner finalizes controller-error ticks before raising. Freeze this
            # session on failure; restart creates a fresh valid runner.
            self.runner.stop("simulation_error")
            self.session = "FINISHED"
            self.clock.reset()
            self._notice = f"{type(error).__name__}: {error}"
            self._refresh_controls()
            messagebox.showerror(self.tr("模拟错误", "Simulation error"), self._notice, parent=self.root)
            return
        self._rate_count += 1
        state = self.runner.engine.state
        if state.terminated:
            self.session = "FINISHED"
            winner = state.winner_id
            self._notice = self.tr(f"对局结束，玩家 {winner + 1} 获胜。", f"Game over. Player {winner + 1} wins.") if winner is not None else self.tr("对局结束。", "Game over.")
        elif self.runner.stopped:
            self.session = "FINISHED"
            self._notice = str(self.runner.stop_reason)
        elif self.human is not None and not bool(state.alive[self.human_seat]):
            self.human.eliminate()
            self.session = "PAUSED"
            self.clock.reset()
            self._notice = self.tr("你的主城已失守。可继续观战或重新开始。", "Your general was captured. Continue spectating or restart.")
            self.spectate_button.pack(fill="x", pady=8, before=self.clock_label)

    def continue_spectating(self):
        if self.mode == "play" and not bool(self.runner.engine.state.alive[self.human_seat]):
            self.spectating = True
            self.spectate_button.pack_forget()
            self.session = "RUNNING" if not self.runner.stopped else "FINISHED"
            self.clock.reset()
            self._notice = self.tr("全图观战中", "Spectating the full map")
            self._refresh()

    def _speed_changed(self, _event=None):
        if self.session == "RUNNING":
            self.clock.accumulate(self.speed)
        self.speed = float(self.speed_var.get().rstrip("×"))
        self.board.focus_set()

    def _frame(self):
        if self.closed:
            return
        dirty = False
        if self.session == "RUNNING":
            self.clock.accumulate(self.speed)
            started = time.monotonic()
            processed = 0
            while self.session == "RUNNING" and self.clock.debt >= 1 and processed < 6:
                if processed and time.monotonic() - started >= .012:
                    break
                self.clock.consume()
                self._advance_tick()
                processed += 1
                dirty = True
        now = time.monotonic()
        if now - self._rate_since >= 1:
            self.actual_rate = self._rate_count / (now - self._rate_since)
            self._rate_count = 0
            self._rate_since = now
            self._refresh_controls()
        if dirty:
            self._refresh()
        self._callback = self.root.after(33, self._frame)

    def _refresh(self):
        player_id = None if self.spectating else self.human_seat
        self._view = snapshot(self.runner.engine.state, player_id=player_id, mode=self.visibility)
        self.board.render(self._view, self.human if self.human_active else None)
        self._refresh_scoreboard()
        self._refresh_controls()

    def _refresh_scoreboard(self):
        view = self._view
        count = self.config.players
        wanted = {str(player) for player in range(count)}
        for item in self.scoreboard.get_children():
            if item not in wanted:
                self.scoreboard.delete(item)
        order = sorted(range(count), key=lambda player: (not bool(view.alive[player]), -int(view.stats.army[player]), player))
        names_cn = {"aggressive": "进攻", "expansion": "扩张", "defensive": "防御", "random": "随机", "human": "你"}
        names_en = {"aggressive": "Attack", "expansion": "Expand", "defensive": "Defend", "random": "Random", "human": "You"}
        for position, player in enumerate(order):
            alive = bool(view.alive[player])
            name = (names_cn if self.chinese else names_en)[self.kinds[player]]
            label = f"P{player + 1}  {name}"
            if not alive:
                label += self.tr(" · 淘汰", " · out")
            values = label, f"{int(view.stats.territory[player]):,}", f"{int(view.stats.army[player]):,}"
            tags = (f"p{player}",) if alive else ("dead",)
            if self.scoreboard.exists(str(player)):
                self.scoreboard.item(str(player), values=values, tags=tags)
                self.scoreboard.move(str(player), "", position)
            else:
                self.scoreboard.insert("", position, iid=str(player), values=values, tags=tags)

    def _refresh_controls(self):
        names = {"PREVIEW": ("准备开始", "Ready to play"), "RUNNING": ("战斗中", "Battle in progress"), "PAUSED": ("已暂停", "Paused"), "FINISHED": ("对局结束", "Game over")}
        self.session_var.set(self.tr(*names[self.session]))
        tick = self.runner.engine.state.tick
        debt = self.clock.debt
        lag = self.tr(f" · 滞后 {debt:.0f} tick", f" · {debt:.0f} ticks behind") if debt >= 2 else ""
        self.clock_var.set(f"Tick {tick:,}  ·  {self.actual_rate:.1f} tick/s{lag}")
        viewpoint = self.tr("全图", "Full map") if self.spectating or self.visibility == "full" else self.tr(f"玩家 {self.human_seat + 1} 视野", f"Player {self.human_seat + 1} view")
        self.header_var.set(f"{self.config.width} × {self.config.height}  ·  {viewpoint}  ·  Seed {self.seed}")
        self.run_button.configure(text=self.tr("暂停", "Pause") if self.session == "RUNNING" else self.tr("开始", "Start") if self.session == "PREVIEW" else self.tr("继续", "Resume"))
        defeated = self.human is not None and not bool(self.runner.engine.state.alive[self.human_seat])
        self.run_button.configure(state="disabled" if self.session == "FINISHED" or defeated else "normal")
        self.step_button.configure(state="normal" if self.session in {"PREVIEW", "PAUSED"} and not defeated else "disabled")
        self.settings_button.configure(state="normal" if self.session == "PREVIEW" else "disabled")
        for button in self.human_buttons:
            button.configure(state="normal" if self.human_active else "disabled")
        human = self.human
        if human is not None:
            cursor = human.cursor if human.selected is not None else None
            coord = "—" if cursor is None else f"({int(cursor) // self.config.width + 1}, {int(cursor) % self.config.width + 1})"
            self.selection_var.set(self.tr(f"规划位置 {coord}  ·  队列 {len(human.queue)}/64", f"Cursor {coord}  ·  Queue {len(human.queue)}/64"))
            self.half_button.configure(text=self.tr("半兵 50% ✓", "Half 50% ✓") if human.half else self.tr("半兵 Z", "Half army Z"))
        else:
            self.selection_var.set(self.tr("观战模式", "Spectator mode"))
        terrain = self.initial.layout.terrain
        structures = self.initial.structure
        mountains = float((terrain == 1).sum()) / self.initial.layout.size
        cities = float((structures == 1).sum()) / self.initial.layout.size
        self.detail_var.set(self.tr(f"山脉 {mountains:.1%}（目标 {self.config.mountain_density:.0%}）\n城市 {cities:.1%}（目标 {self.config.city_density:.0%}）\n主城 / 城市每 2 tick +1\n普通领土每 50 tick +1\n攻占主城即可淘汰对手", f"Mountains {mountains:.1%} (target {self.config.mountain_density:.0%})\nCities {cities:.1%} (target {self.config.city_density:.0%})\nGenerals / cities: +1 / 2 ticks\nPlain land: +1 / 50 ticks\nCapture enemy generals to win"))
        controller_message = human.message if human is not None else ""
        self.message_var.set(self._notice or controller_message or self.tr("暂停时可规划，移动每 tick 执行一步。", "Plan while paused. One move executes per tick."))

    def select_cell(self, cell, *, force=False):
        if self.human_active:
            self._notice = ""
            self.human.select(cell, self._view, force=force)
            self.board.render_queue(self.human)
            self._refresh_controls()

    def human_command(self, command, value=None):
        if not self.human_active:
            return
        self._notice = ""
        if command == "home":
            self.human.select(int(self._view.general_pos), self._view, force=True)
        elif command == "enqueue":
            self.human.enqueue(value, self._view)
        elif command in {"undo", "clear_queue"}:
            getattr(self.human, command)(self._view)
        else:
            getattr(self.human, command)()
        self.board.render_queue(self.human)
        self._refresh_controls()

    def open_settings(self):
        if self.session != "PREVIEW":
            return
        if self._settings_window is not None and self._settings_window.winfo_exists():
            self._settings_window.lift()
            return
        dialog = tk.Toplevel(self.root)
        self._settings_window = dialog
        dialog.title(self.tr("对局设置", "Game settings"))
        dialog.configure(background=PANEL)
        dialog.transient(self.root)
        dialog.resizable(False, False)
        dialog.grab_set()
        frame = ttk.Frame(dialog, padding=20)
        frame.pack(fill="both", expand=True)
        variables = {
            "mode": tk.StringVar(value=self.mode),
            "players": tk.IntVar(value=self.config.players),
            "size": tk.IntVar(value=self.config.width),
            "mountain": tk.IntVar(value=round(self.config.mountain_density * 100)),
            "city": tk.IntVar(value=round(self.config.city_density * 100)),
            "seed": tk.StringVar(value=str(self.seed)),
            "human": tk.IntVar(value=self.human_seat + 1),
            "full": tk.BooleanVar(value=self.visibility == "full"),
        }
        fields = [
            ("模式", "Mode", "mode", ("play", "watch")),
            ("玩家数量", "Players", "players", tuple(range(2, 9))),
            ("地图边长", "Map size", "size", tuple(sorted({15, 25, 35, self.config.width}))),
            ("山脉比例 %", "Mountains %", "mountain", tuple(range(0, 41, 5))),
            ("城市比例 %", "Cities %", "city", tuple(range(0, 21, 5))),
            ("人类席位", "Human seat", "human", tuple(range(1, self.config.players + 1))),
        ]
        widgets = {}
        for row, (cn, en, key, values) in enumerate(fields):
            ttk.Label(frame, text=self.tr(cn, en)).grid(row=row, column=0, sticky="w", padx=(0, 18), pady=5)
            widget = ttk.Combobox(frame, textvariable=variables[key], values=values, state="readonly", width=19)
            widget.grid(row=row, column=1, sticky="ew", pady=5)
            widgets[key] = widget
        ttk.Label(frame, text=self.tr("随机种子", "Seed")).grid(row=6, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=variables["seed"]).grid(row=6, column=1, sticky="ew", pady=5)
        ttk.Checkbutton(frame, text=self.tr("调试：所有玩家全信息", "Debug: full information for all players"), variable=variables["full"]).grid(row=7, column=0, columnspan=2, sticky="w", pady=(8, 12))
        seat_vars, seat_widgets = [], []
        for player in range(8):
            kind = self.kinds[player] if player < len(self.kinds) and self.kinds[player] != "human" else AI_KINDS[player % 4]
            variable = tk.StringVar(value=kind)
            label = ttk.Label(frame, text=f"P{player + 1}")
            widget = ttk.Combobox(frame, textvariable=variable, values=AI_KINDS, state="readonly", width=19)
            label.grid(row=8 + player, column=0, sticky="w", pady=3)
            widget.grid(row=8 + player, column=1, sticky="ew", pady=3)
            seat_vars.append(variable)
            seat_widgets.append((label, widget))

        def update_seats(*_args):
            count = variables["players"].get()
            if variables["human"].get() > count:
                variables["human"].set(count)
            playing = variables["mode"].get() == "play"
            widgets["human"].configure(values=tuple(range(1, count + 1)), state="readonly" if playing else "disabled")
            for player, (label, widget) in enumerate(seat_widgets):
                if player < count:
                    label.grid()
                    widget.grid()
                    is_human = playing and player == variables["human"].get() - 1
                    label.configure(text=f"P{player + 1}" + self.tr("（你）", " (you)") if is_human else f"P{player + 1}")
                    widget.configure(state="disabled" if is_human else "readonly")
                else:
                    label.grid_remove()
                    widget.grid_remove()

        for name in ("players", "human", "mode"):
            variables[name].trace_add("write", update_seats)
        update_seats()

        def apply():
            if self.session != "PREVIEW":
                dialog.destroy()
                return
            try:
                config = MapConfig(width=variables["size"].get(), height=variables["size"].get(), players=variables["players"].get(), mountain_density=variables["mountain"].get() / 100, city_density=variables["city"].get() / 100)
                seed = int(variables["seed"].get())
                if seed < 0:
                    raise ValueError(self.tr("随机种子必须非负", "Seed must be non-negative"))
                # Generate before changing the live session, keeping errors reversible.
                initial = generate_map(config, seed)
                kinds = [variable.get() for variable in seat_vars[:config.players]]
                human_seat = variables["human"].get() - 1
                mode = variables["mode"].get()
                if mode == "play":
                    kinds[human_seat] = "human"
                self.config, self.seed, self.mode = config, seed, mode
                self.human_seat, self.kinds = human_seat, kinds
                self.visibility = "full" if variables["full"].get() else "local"
                self.initial = initial
                self._load_initial()
            except (ValueError, TypeError) as error:
                messagebox.showerror(self.tr("设置无效", "Invalid settings"), str(error), parent=dialog)
                return
            dialog.destroy()

        ttk.Button(frame, text=self.tr("生成预览", "Apply and preview"), style="Accent.TButton", command=apply).grid(row=16, column=0, columnspan=2, sticky="ew", pady=(14, 0))

    def show_help(self):
        self.pause()
        messagebox.showinfo(self.tr("规则与操作", "Rules and controls"), self.tr(
            "目标：攻占其他玩家的主城，成为最后的幸存者。\n图例：王冠为主城，城堡为城市，三角为山脉。\n\n主城与城市每 2 tick 增加 1 兵；普通领土每 50 tick 增加 1 兵。中立城市有 40 兵。只能上下左右移动，不能进入山脉。普通移动留下 1 兵；半兵向下取整。交战兵力按 1:1 抵消，进攻兵力严格大于守军才占领。主城失守则淘汰，剩余领土及减半驻军转交胜方。\n\n左键选择己方格，再点击相邻格规划路线。再次点同一格切换半兵。Shift + 点击强制切换来源。WASD / 方向键移动，Z 半兵，Q 清空队列，E 撤销末步，空格取消选择（保留队列），H 选主城，P 暂停。所有基础行动均有鼠标按钮。\n\n右键 / 中键拖动；滚轮缩放；Home 适应窗口。人类视野为己方领地及周围八格；深色为未知区域。顶部显示种子，设置只在开局预览可修改。单步推进所有玩家的一整个 tick。",
            "Capture enemy generals to become the last surviving player.\nLegend: crown = general, castle = city, triangle = mountain.\n\nGenerals and cities gain 1 army every 2 ticks; plain owned land gains 1 every 50 ticks. Neutral cities start with 40. Move orthogonally; mountains block movement. Normal moves leave 1 army, half moves send floor(army/2). Combat subtracts armies 1:1; attackers must strictly outnumber defenders to capture. Losing a general eliminates that player and transfers their remaining land with halved armies.\n\nSelect tile, then adjacent tiles to queue a path. Click the current tile again for half army. Shift + click forces a new source. WASD / arrows: move. Z: half. Q: clear queue. E: undo last. Space: deselect (keeps queue). H: general. P: pause. Buttons support mouse-only play.\n\nRight / middle drag to pan; wheel to zoom; Home fits the map. Local vision covers owned land and the eight surrounding tiles. Dark cells are fog. Settings are editable in the opening preview. Step advances one complete tick for all players."), parent=self.root)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self._callback is not None:
            self.root.after_cancel(self._callback)
        self.runner.stop("user_stop")
        self.root.destroy()


def launch(config: MapConfig, seed=42, mode="play", human_seat=0,
           visibility="local", kinds=None, speed=1.0):
    """Create the desktop application; headless callers never need this import."""
    root = tk.Tk()
    App(root, config, seed=seed, mode=mode, human_seat=human_seat,
        visibility=visibility, kinds=kinds, speed=speed)
    root.mainloop()
