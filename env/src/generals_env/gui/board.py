"""Canvas-only presentation of immutable game observations.

All map coordinates pass through one transform, shared by drawing and hit tests.
Cell items are allocated once per map, including icons; refreshes update them.
"""

from __future__ import annotations

import math
import tkinter as tk

import numpy as np

PLAYER_COLORS = (
    "#dc5265", "#428cd4", "#36a88b", "#cb983b",
    "#9770d2", "#d17539", "#409daa", "#bd66ac",
)
BOARD_BG = "#101821"
GRID = "#17222e"


class Board(tk.Canvas):
    """Stable vector grid with fog, queue arrows, pan and zoom."""

    def __init__(self, master, *, font_family="TkDefaultFont", **kwargs):
        super().__init__(master, background=BOARD_BG, highlightthickness=0, **kwargs)
        self.font_family = font_family
        self.width = self.height = 0
        self.cell_size = 28.0
        self.offset_x = self.offset_y = 0.0
        self._cells = []
        self._cache = []
        self._view = None
        self._human = None
        self._known_terrain = np.empty(0, dtype=np.uint8)
        self._selection = None
        self._cursor_marker = None
        self._arrows = []
        self._half_labels = []
        self._auto_fit = True
        self.bind("<Configure>", self._resized)

    def reset(self, width: int, height: int):
        self.delete("all")
        self.width, self.height = width, height
        self._cells.clear()
        self._cache = [None] * (width * height)
        self._known_terrain = np.full(width * height, 255, dtype=np.uint8)
        for _ in range(width * height):
            background = self.create_rectangle(0, 0, 0, 0, outline=GRID)
            mountain = self.create_polygon(0, 0, 0, 0, 0, 0, fill="#7b8896", outline="#a4aebb", state="hidden")
            city = self.create_polygon(0, 0, 0, 0, 0, 0, fill="#d2d9df", outline="#ffffff", state="hidden")
            general = self.create_polygon(0, 0, 0, 0, 0, 0, fill="#ffdf83", outline="#fff3c9", state="hidden")
            number = self.create_text(0, 0, fill="white", font=(self.font_family, 11, "bold"))
            self._cells.append((background, mountain, city, general, number))
        self._arrows = [self.create_line(0, 0, 0, 0, fill="#ffe68c", width=3, arrow=tk.LAST, state="hidden") for _ in range(64)]
        self._half_labels = [self.create_text(0, 0, text="50%", fill="#ffffff", state="hidden", font=(self.font_family, 9, "bold")) for _ in range(64)]
        self._selection = self.create_rectangle(0, 0, 0, 0, outline="#ffffff", width=3, state="hidden")
        self._cursor_marker = self.create_rectangle(0, 0, 0, 0, outline="#ffe68c", width=2, dash=(3, 2), state="hidden")
        self._view = None
        self._human = None
        self._auto_fit = True
        self.fit()

    def _resized(self, _event=None):
        if self._auto_fit:
            self.fit()

    def fit(self):
        if not self.width:
            return
        available_w, available_h = max(self.winfo_width(), 100), max(self.winfo_height(), 100)
        self.cell_size = min(70.0, max(6.0, min((available_w - 40) / self.width, (available_h - 40) / self.height)))
        self.offset_x = (available_w - self.width * self.cell_size) / 2
        self.offset_y = (available_h - self.height * self.cell_size) / 2
        self._auto_fit = True
        self._position_items()

    def cell_at(self, x: float, y: float):
        column = math.floor((x - self.offset_x) / self.cell_size)
        row = math.floor((y - self.offset_y) / self.cell_size)
        if 0 <= column < self.width and 0 <= row < self.height:
            return row * self.width + column
        return None

    def center(self, cell: int):
        row, column = divmod(int(cell), self.width)
        return (self.offset_x + (column + 0.5) * self.cell_size,
                self.offset_y + (row + 0.5) * self.cell_size)

    def pan(self, dx: float, dy: float):
        self._auto_fit = False
        self.offset_x += dx
        self.offset_y += dy
        self.move("all", dx, dy)

    def zoom(self, steps: float, x=None, y=None):
        if not self.width:
            return
        x = self.winfo_width() / 2 if x is None else x
        y = self.winfo_height() / 2 if y is None else y
        new_size = min(110.0, max(6.0, self.cell_size * 1.15 ** steps))
        factor = new_size / self.cell_size
        self.offset_x = x - (x - self.offset_x) * factor
        self.offset_y = y - (y - self.offset_y) * factor
        self.cell_size = new_size
        self._auto_fit = False
        self._position_items()

    def _position_items(self):
        size = self.cell_size
        font_size = max(6, min(20, int(size * 0.31)))
        mountain_points = (.19, .77, .48, .23, .82, .77)
        city_points = (.22, .50, .22, .22, .34, .22, .34, .32, .45, .32, .45, .22, .56, .22, .56, .32, .67, .32, .67, .22, .79, .22, .79, .50)
        crown_points = (.20, .23, .36, .34, .50, .15, .64, .34, .80, .23, .71, .51, .29, .51)
        for cell, (background, mountain, city, general, number) in enumerate(self._cells):
            row, column = divmod(cell, self.width)
            x, y = self.offset_x + column * size, self.offset_y + row * size
            self.coords(background, x, y, x + size, y + size)
            for item, points in ((mountain, mountain_points), (city, city_points), (general, crown_points)):
                self.coords(item, *(value * size + (x if i % 2 == 0 else y) for i, value in enumerate(points)))
            self.itemconfigure(number, font=(self.font_family, font_size, "bold"))
            self.coords(number, x + size / 2, y + size * .57)
        self._cache = [None] * len(self._cells)
        if self._view is not None:
            self.render(self._view, self._human)

    def render(self, observation, human=None):
        self._view, self._human = observation, human
        terrain = np.asarray(observation.terrain).reshape(-1)
        structure = np.asarray(observation.structure).reshape(-1)
        owner = np.asarray(observation.owner).reshape(-1)
        army = np.asarray(observation.army).reshape(-1)
        visible = np.asarray(observation.visible).reshape(-1)
        self._known_terrain[visible] = terrain[visible]
        for cell, items in enumerate(self._cells):
            is_visible = bool(visible[cell])
            ground = int(terrain[cell]) if is_visible else int(self._known_terrain[cell])
            building = int(structure[cell]) if is_visible else 0
            player = int(owner[cell]) if is_visible else -3
            count = int(army[cell]) if is_visible else 0
            signature = is_visible, ground, building, player, count
            if signature == self._cache[cell]:
                continue
            self._cache[cell] = signature
            background, mountain, city, general, number = items
            if not is_visible:
                color = "#202b38" if ground != 255 else "#141e2a"
            elif player >= 0:
                color = PLAYER_COLORS[player % len(PLAYER_COLORS)]
            elif ground == 1:
                color = "#34404d"
            elif building == 1:
                color = "#687580"
            else:
                color = "#d3dbe0"
            self.itemconfigure(background, fill=color)
            self.itemconfigure(mountain, state="normal" if ground == 1 else "hidden", fill="#7b8896" if is_visible else "#3f4d5d", outline="#a4aebb" if is_visible else "#465365")
            self.itemconfigure(city, state="normal" if building == 1 and is_visible else "hidden")
            self.itemconfigure(general, state="normal" if building == 2 and is_visible else "hidden")
            # A zero-army owned tile is still owned and must remain visibly so.
            text = self._army_text(count) if is_visible and ground != 1 and (player >= 0 or count > 0 or building > 0) else ""
            self.itemconfigure(number, text=text, fill="#ffffff" if player >= 0 or building else "#243344")
            x, y = self.center(cell)
            self.coords(number, x, y + self.cell_size * (.24 if building else .03))
        self.render_queue(human)

    @staticmethod
    def _army_text(count):
        if count < 10000:
            return str(count)
        if count < 1000000:
            return f"{count / 1000:.0f}k"
        return f"{count / 1000000:.1f}m"

    def render_queue(self, human):
        queue = list(human.queue) if human is not None else []
        for index, item in enumerate(self._arrows):
            if index >= len(queue):
                self.itemconfigure(item, state="hidden")
                self.itemconfigure(self._half_labels[index], state="hidden")
                continue
            action = queue[index]
            source = int(action.source)
            target = source + (-self.width, 1, self.width, -1)[int(action.direction)]
            if not 0 <= source < len(self._cells) or not 0 <= target < len(self._cells):
                self.itemconfigure(item, state="hidden")
                self.itemconfigure(self._half_labels[index], state="hidden")
                continue
            x1, y1 = self.center(source)
            x2, y2 = self.center(target)
            self.coords(item, x1, y1, x2, y2)
            self.itemconfigure(item, state="normal", width=max(2, self.cell_size * .055), arrowshape=(max(5, self.cell_size * .23), max(6, self.cell_size * .28), max(2, self.cell_size * .11)))
            self.coords(self._half_labels[index], (x1 + x2) / 2, (y1 + y2) / 2 - self.cell_size * .15)
            self.itemconfigure(self._half_labels[index], state="normal" if int(action.mode) == 1 else "hidden")
        selected = getattr(human, "selected", None) if human is not None else None
        cursor = getattr(human, "cursor", None) if human is not None else None
        cell = cursor if selected is not None and cursor is not None else selected
        self.itemconfigure(self._cursor_marker, state="hidden")
        if selected is not None and cursor is not None and cursor != selected:
            x, y = self.center(int(selected))
            half = self.cell_size / 2 - 3
            self.coords(self._cursor_marker, x - half, y - half, x + half, y + half)
            self.itemconfigure(self._cursor_marker, state="normal")
        if cell is None or not 0 <= int(cell) < len(self._cells):
            self.itemconfigure(self._selection, state="hidden")
        else:
            x, y = self.center(int(cell))
            half = self.cell_size / 2 - 2
            self.coords(self._selection, x - half, y - half, x + half, y + half)
            self.itemconfigure(self._selection, state="normal", outline="#ffe68c" if human.half else "#ffffff")
