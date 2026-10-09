// Optimistic cursor and queued intent. Only the server changes troop ownership.
export const decode = action => action ? {
  source:Math.floor((action-1)/8), direction:Math.floor((action-1)%8/2), half:!!((action-1)%2),
} : null;
export const targetOf = (action,width) => {
  const move=decode(action);
  return move ? move.source + [-width,1,width,-1][move.direction] : -1;
};
export class MoveInput {
  constructor(changed, send) {
    this.changed=changed;this.send=send;this.reset();
  }
  reset() {
    this.board=null;this.selected=-1;this.half=false;this.base=[];this.pending=[];this.next=1;this.revision=-1;
  }
  observe(view) {
    if(view.revision<this.revision)return false;
    this.revision=view.revision;this.board=view.board;this.base=view.queue||[];
    this.pending=this.pending.filter(c=>c.id>view.last_command_id);
    this.next=Math.max(this.next,view.last_command_id+1);
    return true;
  }
  get queue() {
    const queue=[...this.base];
    for(const command of this.pending) {
      if(command.op==='append')queue.push({id:command.id,action:command.action});
      else if(command.op==='clear')queue.length=0;
      else queue.pop();
    }
    return queue;
  }
  issue(op,action) {
    if(!this.board?.alive[this.board.player_id])return;
    const command={id:this.next++,op};if(action!==undefined)command.action=action;
    this.pending.push(command);this.changed();this.send(command);
  }
  owned(cell) {return cell>=0&&this.board.owner[Math.floor(cell/this.board.width)]?.[cell%this.board.width]===this.board.player_id;}
  move(direction,half=false) {
    const b=this.board,source=this.selected,queue=this.queue;
    if(!b||source<0||!b.alive[b.player_id]||queue.length>=256)return false;
    if(!this.owned(source)&&!queue.some(q=>targetOf(q.action,b.width)===source))return false;
    const x=source%b.width,y=Math.floor(source/b.width),nx=x+[0,1,0,-1][direction],ny=y+[-1,0,1,0][direction];
    if(nx<0||ny<0||nx>=b.width||ny>=b.height||b.terrain[ny][nx]===1)return false;
    const action=1+source*8+direction*2+Number(half||this.half);
    this.selected=ny*b.width+nx;this.half=false;this.issue('append',action);return true;
  }
  select(cell,half=false) {
    if(!this.board)return;
    if(cell===this.selected){this.half=!this.half;this.changed();return;}
    const width=this.board.width,delta=cell-this.selected;
    const adjacent=this.selected>=0&&Math.abs(cell%width-this.selected%width)+Math.abs(Math.floor(cell/width)-Math.floor(this.selected/width))===1;
    if(adjacent&&this.move([-width,1,width,-1].indexOf(delta),half))return;
    if(this.owned(cell)){this.selected=cell;this.half=half;this.changed();}
  }
  clear() {
    const first=this.queue.find(q=>q.action);
    if(first&&this.owned(decode(first.action).source))this.selected=decode(first.action).source;
    this.half=false;this.issue('clear');
  }
  undo() {
    const last=this.queue.at(-1);
    if(!last)return;
    if(last.action)this.selected=decode(last.action).source;
    this.half=false;this.issue('undo');
  }
  deselect(){this.selected=-1;this.half=false;this.changed();}
  home(){this.selected=this.board?.general_pos??-1;this.half=false;this.changed();}
}
