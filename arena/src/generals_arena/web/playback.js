// Every replay tick gets its own paint. A slow display limits playback speed;
// elapsed wall time must never turn into skipped game states.
export class Playback {
  constructor({step, interval, stopped, request = requestAnimationFrame, cancel = cancelAnimationFrame}) {
    Object.assign(this, {step, interval, stopped, request, cancel});
    this.handle = null;
    this.last = null;
    this.running = false;
  }
  start() {
    if (this.running) return;
    this.running = true;
    this.last = null;
    this.handle = this.request(time => this.paint(time));
  }
  stop() {
    this.running = false;
    if (this.handle !== null) this.cancel(this.handle);
    this.handle = this.last = null;
    this.stopped?.();
  }
  paint(time) {
    this.handle = null;
    if (!this.running) return;
    if (this.last === null) this.last = time;
    const interval = this.interval();
    const elapsed = time - this.last;
    if (elapsed >= interval) {
      // Preserve sub-frame timing, but discard the backlog after a stall.
      this.last = time - elapsed % interval;
      if (!this.step()) { this.stop(); return; }
    }
    this.handle = this.request(next => this.paint(next));
  }
}
