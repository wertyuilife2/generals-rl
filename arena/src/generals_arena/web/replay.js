// Retain every tick. Bounded snapshots make repeated seeks independent of
// how far the target is from the beginning of a long match.
// Only these three grids are edited in place. Terrain/visibility stay fixed;
// tick metadata is replaced with immutable values from the recorded frames.
const copyBoard = board => ({...board, ...Object.fromEntries(
  ['owner','army','structure'].map(key=>[key,board[key].map(row=>row.slice())]),
)});
export class ReplayTimeline {
  constructor(frames, {interval=128, capacity=32}={}) {
    this.frames = frames;
    this.interval = interval;
    this.capacity = Math.max(2, capacity);
    this.snapshots = new Map([[0, copyBoard(frames[0])]]);
    this.index = 0;
    this.board = copyBoard(frames[0]);
  }
  seek(target) {
    target = Math.max(0, Math.min(this.frames.length-1, Math.floor(target)));
    let start = this.index <= target ? this.index : 0;
    for (const index of this.snapshots.keys()) {
      if (index > start && index <= target) start = index;
    }
    if (this.index > target || start > this.index) {
      this.board = copyBoard(this.snapshots.get(start));
      this.index = start;
    }
    while (this.index < target) {
      const frame = this.frames[++this.index], board = this.board;
      for (const [cell,owner,army,structure] of frame.cells) {
        const y = Math.floor(cell/board.width), x = cell%board.width;
        board.owner[y][x] = owner; board.army[y][x] = army; board.structure[y][x] = structure;
      }
      board.tick = frame.tick; board.alive = frame.alive;
      board.stats = frame.stats; board.forfeited = frame.forfeited;
      if (this.index % this.interval === 0 && !this.snapshots.has(this.index)) {
        this.snapshots.set(this.index, copyBoard(board));
        if (this.snapshots.size > this.capacity) this.snapshots.delete([...this.snapshots.keys()][1]);
      }
    }
    return this.board;
  }
}
