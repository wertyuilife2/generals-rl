import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Playback } from '../../src/generals_arena/web/playback.js';

function clock(speed, length=10000) {
  let callback, handle = 0, frame = 0, stopped = 0;
  const painted = [];
  const playback = new Playback({
    request: cb => { callback=cb; return ++handle; },
    cancel: () => { callback=null; },
    interval: () => 500/speed,
    step: () => { painted.push(++frame); return frame<length; },
    stopped: () => stopped++,
  });
  return {playback, painted, stopped:()=>stopped, paint:time=>callback?.(time)};
}
test('25x paints each tick separately at 50 ticks/sec on a 60 Hz display',()=>{
  const c=clock(25); c.playback.start();
  for(let i=0;i<=600;i++) c.paint(i*1000/60);
  assert.equal(c.painted.length,500);
  assert.deepEqual(c.painted,Array.from({length:500},(_,i)=>i+1));
});
test('stall / background tab and slow displays never skip replay ticks',()=>{
  const c=clock(25);c.playback.start();c.paint(0);c.paint(10000);
  assert.deepEqual(c.painted,[1]);
  for(let i=1;i<=60;i++)c.paint(10000+i*50);
  assert.deepEqual(c.painted,Array.from({length:61},(_,i)=>i+1));
});
test('pause/resume has no backlog and the last tick stops playback',()=>{
  const c=clock(1,2);c.playback.start();c.paint(0);c.paint(500);
  c.playback.stop();c.paint(10000);assert.deepEqual(c.painted,[1]);
  c.playback.start();c.paint(15000);assert.deepEqual(c.painted,[1]);
  c.paint(15500);assert.deepEqual(c.painted,[1,2]);
  assert.equal(c.playback.running,false);assert.equal(c.stopped(),2);
});
