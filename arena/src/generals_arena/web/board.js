// Presentation adapted from bzy-nya.github.io/gen/js/renderer.js.
// Uses server observations; no game rules or AI execution live in the browser.
import { SVGIcons } from './svgIcons.js';
import { decode } from './input.js';

export const COLORS = ['rgba(255,0,0,.9)', 'rgba(39,146,255,.9)', 'rgba(0,128,0,.9)',
  'rgba(0,128,128,.9)', 'rgba(250,140,1,.9)', 'rgba(240,50,230,.9)',
  'rgba(128,0,128,.9)', 'rgba(155,1,1,.9)'];

export class Battlefield {
  constructor(canvas, onSelect) {
    this.canvas = canvas;
    this.onSelect = onSelect;
    this.board = null;
    this.zoom = 1;
    this.offsetX = 0;
    this.offsetY = 0;
    this.cellSize = 24;
    this.initialized = false;
    this.icons = {};
    for (const [name, svg] of Object.entries(SVGIcons)) {
      const icon = new Image();
      icon.onload = () => {this.viewKey=null;this.draw();};
      icon.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
      this.icons[name] = icon;
    }
    canvas.addEventListener('contextmenu', e => e.preventDefault());
    canvas.addEventListener('pointerdown', e => {
      if (!this.board) return;
      this.pointer = {id:e.pointerId, x:e.clientX, y:e.clientY, startX:e.clientX, startY:e.clientY, dragged:false, button:e.button};
      canvas.setPointerCapture(e.pointerId);
      canvas.focus();
    });
    canvas.addEventListener('pointermove', e => {
      if (!this.pointer || this.pointer.id !== e.pointerId) return;
      const p = this.pointer;
      if (Math.hypot(e.clientX-p.startX, e.clientY-p.startY) > 5) p.dragged = true;
      if (p.dragged) {
        this.offsetX += e.clientX-p.x;
        this.offsetY += e.clientY-p.y;
        this.draw();
      }
      p.x = e.clientX; p.y = e.clientY;
    });
    canvas.addEventListener('pointerup', e => {
      if (!this.pointer || this.pointer.id !== e.pointerId) return;
      const p = this.pointer;
      this.pointer = null;
      if (!p.dragged && p.button === 0) {
        const rect = canvas.getBoundingClientRect();
        const x = Math.floor((e.clientX-rect.left-this.offsetX)/this.cellSize);
        const y = Math.floor((e.clientY-rect.top-this.offsetY)/this.cellSize);
        if (x>=0 && y>=0 && x<this.board.width && y<this.board.height) this.onSelect(x,y,e.shiftKey);
      }
    });
    canvas.addEventListener('pointercancel', () => {this.pointer=null;});
    canvas.addEventListener('wheel', e => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      this.adjustZoom(e.deltaY < 0 ? 1.15 : 1/1.15, e.clientX-rect.left, e.clientY-rect.top);
    }, {passive:false});
    this.observer = new ResizeObserver(() => {this.initialized=false; this.draw();});
    this.observer.observe(canvas);
  }
  render(board, selected=-1, overlay={}) {
    if (this.board && (this.board.width!==board.width || this.board.height!==board.height ||
        (this.board.player_id>=0)!==(board.player_id>=0))) this.initialized=false;
    this.board=board; this.selected=selected; this.overlay=overlay; this.draw();
  }
  reset() {this.zoom=1;this.initialized=false;this.draw();}
  adjustZoom(factor, x=this.canvas.clientWidth/2, y=this.canvas.clientHeight/2) {
    const old=this.zoom;
    this.zoom=Math.max(.35,Math.min(5,this.zoom*factor));
    const actual=this.zoom/old;
    this.offsetX=x-(x-this.offsetX)*actual;
    this.offsetY=y-(y-this.offsetY)*actual;
    this.draw();
  }
  draw() {
    if (!this.board || !this.canvas.clientWidth || !this.canvas.clientHeight) return;
    const canvas=this.canvas, ctx=canvas.getContext('2d'), ratio=window.devicePixelRatio||1;
    const width=canvas.clientWidth, height=canvas.clientHeight;
    if (canvas.width!==Math.round(width*ratio) || canvas.height!==Math.round(height*ratio)) {
      canvas.width=Math.round(width*ratio);canvas.height=Math.round(height*ratio);
    }
    ctx.setTransform(ratio,0,0,ratio,0,0);
    const b=this.board, human=b.player_id>=0, bottom=human?30:130;
    const aside=width>760 ? 245 : 0, usableW=Math.max(200,width-aside-60), usableH=Math.max(140,height-80-bottom);
    this.cellSize=Math.min(48,usableW/b.width,usableH/b.height)*this.zoom;
    const size=this.cellSize;
    if (!this.initialized) {
      this.offsetX=(width-aside-b.width*size)/2;
      this.offsetY=(height+30-bottom-b.height*size)/2;
      this.initialized=true;
    }
    const key=[width,height,ratio,b.width,b.height,size,this.offsetX,this.offsetY].join(',');
    const full=this.viewKey!==key;
    if(full){
      ctx.fillStyle='#111';ctx.fillRect(0,0,width,height);
      this.viewKey=key;this.cells=new Float64Array(b.width*b.height*5).fill(NaN);
    }
    const overlays=new Map();
    for(const item of this.overlay?.queue||[]) {
      const move=decode(item.action);if(!move)continue;
      const list=overlays.get(move.source)||[];list.push(move);overlays.set(move.source,list);
    }
    const overlayKeys=new Map([...overlays].map(([cell,moves])=>[cell,moves.map(m=>`${m.direction}:${+m.half}`).join(',')]));
    if(this.selected>=0)overlayKeys.set(this.selected,(overlayKeys.get(this.selected)||'')+`s${+!!this.overlay?.half}`);
    this.lastPaintedCells=0;
    for (let y=0;y<b.height;y++) for (let x=0;x<b.width;x++) {
      const px=this.offsetX+x*size,py=this.offsetY+y*size;
      if (px+size<0 || py+size<0 || px>width || py>height) continue;
      const owner=b.owner[y][x],terrain=b.terrain[y][x],structure=b.structure[y][x],army=b.army[y][x];
      const fog=owner===-3, obstacle=Number(fog && !!b.fog_obstacles?.[y]?.[x]);
      const cell=y*b.width+x,index=cell*5;
      if(!full&&this.cells[index]===owner&&this.cells[index+1]===terrain&&this.cells[index+2]===structure&&this.cells[index+3]===army&&this.cells[index+4]===obstacle&&overlayKeys.get(cell)===this.overlayKeys?.get(cell))continue;
      this.cells[index]=owner;this.cells[index+1]=terrain;this.cells[index+2]=structure;this.cells[index+3]=army;
      this.cells[index+4]=obstacle;
      this.lastPaintedCells++;
      // Clear the old tile before drawing translucent player colors.
      ctx.fillStyle='#111';ctx.fillRect(px,py,size,size);
      ctx.fillStyle=fog?(obstacle?'#555':'#252525'):owner>=0?COLORS[owner]:terrain===1?'#acacac':structure===1?'#555':'#ddd';
      ctx.fillRect(px,py,size,size);
      ctx.strokeStyle='#111';ctx.lineWidth=Math.max(1,size/26);ctx.strokeRect(px,py,size,size);
      const icon=this.icons[obstacle||terrain===1?'mountain':structure===2?'general':structure===1?'city':''];
      if ((!fog || obstacle) && icon?.complete && icon.naturalWidth) ctx.drawImage(icon,px+size*.07,py+size*.07,size*.86,size*.86);
      if (!fog && terrain!==1 && army>0) {
        ctx.font=`${Math.max(9,size/(army>=1000?2.8:2.2))}px Arial`;
        ctx.fillStyle='#fff';ctx.textAlign='center';ctx.textBaseline='middle';
        const text=army>=100000?`${Math.round(army/1000)}k`:String(army);
        ctx.strokeStyle='rgba(0,0,0,.8)';ctx.lineWidth=2;ctx.strokeText(text,px+size/2,py+size/2);
        ctx.fillText(text,px+size/2,py+size/2);
      }
      for(const move of overlays.get(cell)||[]) {
        const dx=[0,1,0,-1][move.direction],dy=[-1,0,1,0][move.direction];
        const cx=px+size*.5,cy=py+size*.5,tipX=cx+dx*size*.42,tipY=cy+dy*size*.42;
        ctx.strokeStyle=move.half?'#ffd65c':'#fff';ctx.lineWidth=Math.max(2,size*.055);ctx.beginPath();
        ctx.moveTo(cx+dx*size*.1,cy+dy*size*.1);ctx.lineTo(tipX,tipY);
        ctx.moveTo(tipX-dx*size*.17-dy*size*.12,tipY-dy*size*.17+dx*size*.12);
        ctx.lineTo(tipX,tipY);ctx.lineTo(tipX-dx*size*.17+dy*size*.12,tipY-dy*size*.17-dx*size*.12);ctx.stroke();
      }
      if (cell===this.selected) {
        ctx.strokeStyle='#fff';ctx.lineWidth=2.5;ctx.strokeRect(px+2,py+2,size-4,size-4);
        if(this.overlay?.half){ctx.font=`bold ${Math.max(9,size*.25)}px Arial`;ctx.fillStyle='#ffd65c';ctx.textAlign='right';ctx.textBaseline='top';ctx.fillText('50%',px+size-3,py+3);}
      }
    }
    this.overlayKeys=overlayKeys;
  }
}
