"""Cross-platform mouse and keyboard normalization, independent of game rules."""

from __future__ import annotations


def wheel_steps(event, window_system="x11"):
    """Normalize X11 button events and Windows/macOS wheel deltas."""
    if window_system == "x11":
        return 1 if getattr(event, "num", None) == 4 else -1 if getattr(event, "num", None) == 5 else 0
    delta = getattr(event, "delta", 0)
    if not delta:
        return 0
    return max(-4.0, min(4.0, delta / 120)) if abs(delta) >= 120 else (1 if delta > 0 else -1)


class InputBindings:
    """Calls App commands for both keys and buttons; keeps no held-key state."""

    def __init__(self, app):
        self.app = app
        self.board = app.board
        self.drag_anchor = None
        self.window_system = str(app.root.tk.call("tk", "windowingsystem"))
        self.board.bind("<Button-1>", self.click)
        for button in (2, 3):
            self.board.bind(f"<ButtonPress-{button}>", self.drag_start)
            self.board.bind(f"<B{button}-Motion>", self.drag)
            self.board.bind(f"<ButtonRelease-{button}>", self.drag_end)
        if self.window_system == "x11":
            self.board.bind("<Button-4>", self.wheel)
            self.board.bind("<Button-5>", self.wheel)
        else:
            self.board.bind("<MouseWheel>", self.wheel)
        app.root.bind("<KeyPress>", self.key)
        app.root.bind("<FocusOut>", self.focus_out, add="+")

    def click(self, event):
        self.board.focus_set()
        cell = self.board.cell_at(event.x, event.y)
        if cell is not None:
            self.app.select_cell(cell, force=bool(event.state & 0x1))

    def drag_start(self, event):
        self.board.focus_set()
        self.drag_anchor = event.x, event.y
        self.board.configure(cursor="fleur")

    def drag(self, event):
        if self.drag_anchor is not None:
            self.board.pan(event.x - self.drag_anchor[0], event.y - self.drag_anchor[1])
            self.drag_anchor = event.x, event.y

    def drag_end(self, _event=None):
        self.drag_anchor = None
        self.board.configure(cursor="")

    def wheel(self, event):
        self.board.zoom(wheel_steps(event, self.window_system), event.x, event.y)
        return "break"

    def key(self, event):
        focused = self.app.root.focus_get()
        if focused is not None and focused.winfo_class() in {"Entry", "TEntry", "TCombobox", "Spinbox", "TSpinbox", "Text", "Scale", "TScale"}:
            return None
        key = event.keysym.lower()
        directions = {"w": 0, "up": 0, "d": 1, "right": 1, "s": 2, "down": 2, "a": 3, "left": 3}
        if key in directions:
            self.app.human_command("enqueue", directions[key])
        elif key in {"q", "e", "z", "space", "h"}:
            self.app.human_command({"q": "clear_queue", "e": "undo", "z": "toggle_half", "space": "deselect", "h": "home"}[key])
        elif key == "p":
            self.app.toggle_running()
        elif key == "home":
            self.board.fit()
        elif key in {"plus", "equal", "kp_add"}:
            self.board.zoom(1)
        elif key in {"minus", "kp_subtract"}:
            self.board.zoom(-1)
        else:
            return None
        return "break"

    def focus_out(self, _event=None):
        self.drag_end()
        # FocusOut also fires when moving between widgets within our window.
        self.app.root.after_idle(self._check_focus)

    def _check_focus(self):
        if self.app.closed:
            return
        try:
            focused = self.app.root.focus_displayof()
        except Exception:
            focused = None
        if focused is None and self.app.human_active and self.app.session == "RUNNING":
            self.app.pause(self.app.tr("窗口失去焦点，已暂停", "Paused while window is unfocused"))
