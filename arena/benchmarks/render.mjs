// Count real replay tiles repainted by the renderer. This measures rendering
// work avoided, not GPU performance or browser FPS; no browser is automated.
import {readFileSync} from 'node:fs';
import {Battlefield} from '../src/generals_arena/web/board.js';
if(!process.argv[2])throw new Error('Usage: node arena/benchmarks/render.mjs frames.json');
globalThis.window={devicePixelRatio:2};
globalThis.ResizeObserver=class{observe(){}};
globalThis.Image=class{complete=true;naturalWidth=100;};
const context=new Proxy({},{get:()=>()=>{}});
const canvas={clientWidth:1600,clientHeight:1000,width:0,height:0,getContext:()=>context,addEventListener:()=>{}};
const renderer=new Battlefield(canvas,()=>{}),frames=JSON.parse(readFileSync(process.argv[2],'utf8'));
const board=structuredClone(frames[0]);let painted=0;
for(const frame of frames){
  for(const [cell,owner,army,structure] of frame.cells||[]){
    const y=Math.floor(cell/board.width),x=cell%board.width;
    Object.assign(board.owner[y],{[x]:owner});board.army[y][x]=army;board.structure[y][x]=structure;
  }
  renderer.render(board);painted+=renderer.lastPaintedCells;
}
const full=frames.length*board.width*board.height;
console.log(JSON.stringify({frames:frames.length,full_redraw_tiles:full,dirty_redraw_tiles:painted,
  avoided_percent:100*(1-painted/full)},null,2));
