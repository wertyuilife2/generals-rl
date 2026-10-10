import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Playback } from '../../src/generals_arena/web/playback.js';

test('browser animation APIs retain the Window receiver when playback starts and pauses',()=>{
  const request=globalThis.requestAnimationFrame,cancel=globalThis.cancelAnimationFrame;
  let callback,frames=0,cancelled=0;
  globalThis.requestAnimationFrame=function(cb){
    assert.equal(this,globalThis,'browser requestAnimationFrame requires its Window receiver');
    callback=cb;return 17;
  };
  globalThis.cancelAnimationFrame=function(handle){
    assert.equal(this,globalThis,'browser cancelAnimationFrame requires its Window receiver');
    assert.equal(handle,17);cancelled++;
  };
  try {
    const playback=new Playback({step:()=>{frames++;return true;},interval:()=>100});
    playback.start();callback(0);callback(100);
    assert.equal(frames,1);
    playback.stop();assert.equal(cancelled,1);
    playback.start();callback(200);callback(300);
    assert.equal(frames,2);playback.stop();
  } finally {
    if(request===undefined)delete globalThis.requestAnimationFrame;else globalThis.requestAnimationFrame=request;
    if(cancel===undefined)delete globalThis.cancelAnimationFrame;else globalThis.cancelAnimationFrame=cancel;
  }
});

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
