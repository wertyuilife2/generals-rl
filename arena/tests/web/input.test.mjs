import {test} from 'node:test';
import assert from 'node:assert/strict';
import {MoveInput,decode} from '../../src/generals_arena/web/input.js';
function fixture() {
  const sent=[],grid=v=>Array.from({length:5},()=>Array(5).fill(v));
  const board={width:5,height:5,player_id:0,general_pos:12,owner:grid(-1),terrain:grid(0),alive:[true,true]};
  board.owner[2][2]=0;
  const input=new MoveInput(()=>{},c=>sent.push(c));
  input.observe({board,revision:0,last_command_id:0,queue:[]});input.home();
  return {input,sent,board};
}
test('rapid movement builds a path through future captures before any network reply',()=>{
  const {input,sent}=fixture();
  assert.equal(input.move(1),true);assert.equal(input.move(2),true);assert.equal(input.move(3),true);
  assert.equal(input.selected,17);assert.deepEqual(sent.map(c=>decode(c.action).source),[12,13,18]);
  assert.deepEqual(sent.map(c=>c.id),[1,2,3]);assert.equal(input.queue.length,3);
});
test('half army resets after a move; undo and clear are queued in order',()=>{
  const {input,sent}=fixture();input.select(12);assert.equal(input.half,true);
  input.move(1);input.move(2);assert.deepEqual(sent.map(c=>decode(c.action).half),[true,false]);
  input.undo();assert.equal(input.selected,13);assert.equal(input.queue.length,1);
  input.clear();assert.equal(input.selected,12);assert.equal(input.queue.length,0);
  assert.deepEqual(sent.map(c=>c.op),['append','append','undo','clear']);
  input.deselect();assert.equal(input.selected,-1);assert.equal(input.move(0),false);
});
test('server acknowledgement merges pending intent without moving the cursor backward',()=>{
  const {input,board}=fixture();input.move(1);input.move(2);
  const first=input.queue[0];
  input.observe({board,revision:1,last_command_id:1,queue:[first]});
  assert.equal(input.queue.length,2);assert.equal(input.selected,18);
  board.tick=1;board.owner[2][3]=0;
  input.observe({board,revision:2,last_command_id:1,queue:[]});
  assert.equal(input.queue.length,1);assert.equal(input.selected,18);
  assert.equal(input.observe({board,revision:0,last_command_id:0,queue:[]}),false);
});
test('known mountains and wrapping over a board edge cannot be queued',()=>{
  const {input,board}=fixture();board.terrain[2][3]=1;
  assert.equal(input.move(1),false);assert.equal(input.selected,12);
  input.selected=4;board.owner[0][4]=0;assert.equal(input.move(1),false);
});

test('a mountain-shaped fog marker may be a city, so movement into it remains possible',()=>{
  const {input,board,sent}=fixture();
  board.terrain[2][3]=255;board.fog_obstacles=Array.from({length:5},()=>Array(5).fill(false));board.fog_obstacles[2][3]=true;
  assert.equal(input.move(1),true);assert.equal(sent.length,1);
});

test('surrender clears pending moves and prevents further commands',()=>{
  const {input,sent,board}=fixture();input.move(1);input.issue('surrender');
  assert.equal(input.queue.length,0);input.issue('append',0);
  assert.deepEqual(sent.map(c=>c.op),['append','surrender']);
  board.forfeited=[0];input.observe({board,revision:1,last_command_id:2,queue:[]});
  input.issue('append',0);assert.equal(sent.length,2);
});
