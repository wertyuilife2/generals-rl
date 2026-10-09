// Pure unit tests for the server-board renderer's input coordinate boundary.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { Battlefield } from '../../src/generals_arena/web/board.js';

globalThis.window = { devicePixelRatio: 2 };
globalThis.ResizeObserver = class { constructor(fn) {this.fn=fn;} observe() {} };
globalThis.Image = class { complete=true; naturalWidth=100; set src(value) {this.value=value;} };

function fixture() {
  const listeners={}, texts=[], images=[];
  const context=new Proxy({}, {get(_,key) {
    if(key==='fillText')return (...args)=>texts.push(args);
    if(key==='drawImage')return (...args)=>images.push(args);
    return ()=>{};
  }});
  const canvas={clientWidth:1000,clientHeight:800,width:0,height:0,
    addEventListener:(key,fn)=>{listeners[key]=fn;},getContext:()=>context,
    getBoundingClientRect:()=>({left:100,top:50}),setPointerCapture:()=>{},focus:()=>{}};
  const selected=[];
  const renderer=new Battlefield(canvas,(x,y,half)=>selected.push([x,y,half]));
  const grid=value=>Array.from({length:10},()=>Array(10).fill(value));
  renderer.render({width:10,height:10,owner:grid(-1),terrain:grid(0),structure:grid(0),army:grid(0)});
  const pointer=(x,y,shiftKey=false)=>({pointerId:1,button:0,clientX:x,clientY:y,shiftKey});
  return {renderer,canvas,listeners,selected,pointer,texts,images};
}

test('click is mapped from CSS pixels, independent of retina backing size',()=>{
  const f=fixture();
  assert.equal(f.canvas.width,2000);
  assert.equal(f.renderer.cellSize,48);
  // Board starts at CSS (137.5,110), plus the canvas offset (100,50).
  const click=f.pointer(100+137.5+2.5*48,50+110+3.5*48,true);
  f.listeners.pointerdown(click);f.listeners.pointerup(click);
  assert.deepEqual(f.selected,[[2,3,true]]);
});

test('zoom preserves pointer anchor, drag pans without also issuing a move',()=>{
  const f=fixture(),r=f.renderer;
  const before=[(350-r.offsetX)/r.cellSize,(300-r.offsetY)/r.cellSize];
  r.adjustZoom(1.5,350,300);
  assert.ok(Math.abs((350-r.offsetX)/r.cellSize-before[0])<1e-9);
  assert.ok(Math.abs((300-r.offsetY)/r.cellSize-before[1])<1e-9);
  const old=r.offsetX;
  f.listeners.pointerdown(f.pointer(400,350));
  f.listeners.pointermove(f.pointer(500,400));
  f.listeners.pointerup(f.pointer(500,400));
  assert.equal(r.offsetX,old+100);
  assert.deepEqual(f.selected,[]);
});

test('hidden cells reveal neither structure icons nor army labels',()=>{
  const f=fixture();
  f.renderer.board.owner[0][0]=-3;
  f.renderer.board.structure[0][0]=2;
  f.renderer.board.army[0][0]=98765;
  f.renderer.draw();
  assert.equal(f.images.length,0);
  assert.equal(f.texts.length,0);
});

test('unchanged ticks repaint nothing; an army update repaints only its tile',()=>{
  const f=fixture(),r=f.renderer;
  assert.equal(r.lastPaintedCells,100);
  r.render(r.board);assert.equal(r.lastPaintedCells,0);
  r.board.army[2][3]=10;r.render(r.board);
  assert.equal(r.lastPaintedCells,1);
  assert.equal(f.texts.at(-1)[0],'10');
});

test('selection and queue arrows erase their old cells without a full redraw',()=>{
  const f=fixture(),r=f.renderer,b=r.board;
  r.render(b,13,{queue:[{action:1+12*8+1*2}]});assert.equal(r.lastPaintedCells,2);
  r.render(b,14,{queue:[]});assert.equal(r.lastPaintedCells,3);
  r.reset();assert.equal(r.lastPaintedCells,0); // Same transform needs no repaint.
  r.adjustZoom(1.1);assert.ok(r.lastPaintedCells>50);
});
