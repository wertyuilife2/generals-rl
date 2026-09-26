"""Desktop smoke checks run on an available display; never require one headlessly."""
import types
import unittest
from unittest.mock import patch

import numpy as np

try:
    import tkinter as tk
    from generals_env.gui import App
    from generals_env.gui.app import RealtimeClock
    from generals_env.gui.input import wheel_steps
except ImportError as error:
    raise unittest.SkipTest(f"Tk unavailable: {error}")

from generals_env.config import MapConfig


class GuiUtilityTests(unittest.TestCase):
    def test_bounded_catchup_preserves_complete_tick_debt(self):
        clock = RealtimeClock()
        clock.reset(now=0)
        clock.accumulate(speed=1, now=10)
        for _ in range(6):
            self.assertTrue(clock.consume())
        self.assertEqual(clock.debt, 14)
        # Pause/resume discards paused wall time, not by running stale ticks later.
        clock.reset(now=100)
        clock.accumulate(speed=2, now=100.25)
        self.assertEqual(clock.debt, 1)
        self.assertTrue(clock.consume())
        self.assertFalse(clock.consume())

    def test_wheel_events_are_normalized_for_x11_and_windows(self):
        self.assertEqual(wheel_steps(types.SimpleNamespace(num=4), "x11"), 1)
        self.assertEqual(wheel_steps(types.SimpleNamespace(num=5), "x11"), -1)
        self.assertEqual(wheel_steps(types.SimpleNamespace(delta=120), "win32"), 1)
        self.assertEqual(wheel_steps(types.SimpleNamespace(delta=-240), "win32"), -2)
        self.assertEqual(wheel_steps(types.SimpleNamespace(delta=1), "aqua"), 1)
        self.assertEqual(wheel_steps(types.SimpleNamespace(delta=0), "win32"), 0)


class DesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            probe = tk.Tk()
            probe.destroy()
        except tk.TclError as error:
            raise unittest.SkipTest(f"No desktop display: {error}")

    def setUp(self):
        self.root = tk.Tk()
        self.app = App(self.root, MapConfig(width=15, height=15, players=4,
                                         mountain_density=0, city_density=0), seed=41)
        self.root.update()

    def tearDown(self):
        self.app.close()

    def click(self, cell):
        x, y = self.app.board.center(cell)
        self.app.board.event_generate("<Button-1>", x=round(x), y=round(y))
        self.root.update_idletasks()

    def path(self):
        source = int(self.app._view.general_pos)
        y, x = divmod(source, self.app.config.width)
        delta = 1 if x < self.app.config.width - 1 else -1
        return source, source + delta

    def test_mouse_queue_half_undo_clear_and_deselect(self):
        source, target = self.path()
        self.click(source)
        self.assertEqual(self.app.human.selected, source)
        self.app.half_button.invoke()
        self.assertTrue(self.app.human.half)
        self.click(target)
        self.assertEqual(len(self.app.human.queue), 1)
        self.assertEqual(int(self.app.human.queue[0].mode), 1)
        self.assertFalse(self.app.human.half)
        # Clicking the visible planning cursor selects half for the next order.
        self.click(target)
        self.assertTrue(self.app.human.half)
        self.app.human_buttons[2].invoke()  # undo
        self.assertEqual(len(self.app.human.queue), 0)
        self.click(target)
        self.app.human_buttons[4].invoke()  # deselect keeps queued action
        self.assertIsNone(self.app.human.selected)
        self.assertEqual(len(self.app.human.queue), 1)
        self.app.human_buttons[3].invoke()  # clear
        self.assertEqual(len(self.app.human.queue), 0)
        self.app.human_buttons[0].invoke()  # general
        self.assertEqual(self.app.human.selected, source)

    def test_hit_testing_and_stable_items_survive_pan_zoom(self):
        board = self.app.board
        ids = board.find_all()
        cells = (0, 45, 224)
        board.pan(34, -17)
        for cell in cells:
            self.assertEqual(board.cell_at(*board.center(cell)), cell)
        board.zoom(2, *board.center(45))
        for cell in cells:
            self.assertEqual(board.cell_at(*board.center(cell)), cell)
        self.app.single_step()
        self.assertEqual(ids, board.find_all())
        board.fit()
        self.assertIsNone(board.cell_at(board.offset_x - 1, board.offset_y))

    def test_pause_step_restart_and_configuration_lock(self):
        self.app.single_step()
        self.assertEqual(self.app.session, "PAUSED")
        self.assertEqual(self.app.runner.engine.state.tick, 1)
        self.assertEqual(str(self.app.settings_button.cget("state")), "disabled")
        self.app.toggle_running()
        self.assertEqual(self.app.session, "RUNNING")
        self.app.pause()
        state = self.app.runner.engine.state.copy()
        self.root.update()
        np.testing.assert_array_equal(state.army, self.app.runner.engine.state.army)
        self.app.restart()
        self.assertEqual(self.app.session, "PREVIEW")
        self.assertEqual(self.app.runner.engine.state.tick, 0)
        self.assertEqual(str(self.app.settings_button.cget("state")), "normal")
        for _ in range(20):
            self.app.single_step()
        first = self.app.runner.engine.state.copy()
        self.app.restart()
        for _ in range(20):
            self.app.single_step()
        np.testing.assert_array_equal(first.owner, self.app.runner.engine.state.owner)
        np.testing.assert_array_equal(first.army, self.app.runner.engine.state.army)

    def test_settings_dialog_and_all_supported_player_counts(self):
        self.app.open_settings()
        self.root.update()
        self.assertTrue(self.app._settings_window.winfo_exists())
        self.assertEqual(self.root.grab_current(), self.app._settings_window)
        self.app._settings_window.destroy()
        for players in range(2, 9):
            self.app.config = MapConfig(width=15, height=15, players=players)
            self.app.kinds = ["random"] * players
            self.app._create_initial()
            self.app._load_initial()
            self.assertEqual(len(self.app.scoreboard.get_children()), players)
        for size in (25, 35):
            self.app.config = MapConfig(width=size, height=size, players=8)
            self.app._create_initial()
            self.app._load_initial()
            self.app.single_step()
            self.assertEqual(len(self.app.board._cells), size * size)

    def test_controller_error_stops_single_step_and_disables_resume(self):
        class BrokenController:
            def act(self, _obs):
                raise RuntimeError("test policy failure")
            def on_result(self, _result):
                pass

        self.app.runner.controllers[1] = BrokenController()
        with patch("generals_env.gui.app.messagebox.showerror") as report:
            self.app.single_step()
        report.assert_called_once()
        self.assertEqual(self.app.session, "FINISHED")
        self.assertTrue(self.app.runner.stopped)
        self.assertFalse(self.app.human_active)
        self.assertEqual(str(self.app.run_button.cget("state")), "disabled")
        self.assertIn("test policy failure", self.app.message_var.get())
        self.app.toggle_running()
        self.assertEqual(self.app.session, "FINISHED")
        self.app.restart()
        self.assertEqual(self.app.session, "PREVIEW")

    def test_human_defeat_pauses_before_spectating_same_match(self):
        # A scripted capture uses the normal Runner; no GUI code edits game arrays.
        state = self.app.runner.engine.state
        source = int(state.general_pos[0])
        y, x = divmod(source, state.layout.width)
        attacker = source + (1 if x < state.layout.width - 1 else -1)
        state.owner.flat[attacker] = 1
        state.army.flat[attacker] = 20
        from generals_env.state import Action, Direction
        direction = Direction.LEFT if attacker > source else Direction.RIGHT

        class AttackController:
            def act(self, _obs):
                return Action.move(attacker, direction)
            def on_result(self, _result):
                pass

        self.app.runner.controllers[1] = AttackController()
        self.app.single_step()
        self.assertEqual(self.app.session, "PAUSED")
        self.assertFalse(self.app.human_active)
        old_runner = self.app.runner
        old_tick = state.tick
        self.app.single_step()
        self.assertEqual(state.tick, old_tick)
        self.app.continue_spectating()
        self.assertIs(self.app.runner, old_runner)
        self.assertEqual(self.app.session, "RUNNING")
        self.assertEqual(self.app._view.player_id, -1)
        self.assertTrue(self.app._view.visible.all())


if __name__ == "__main__":
    unittest.main()
