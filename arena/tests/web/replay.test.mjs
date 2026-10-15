import assert from 'node:assert/strict';
import { test } from 'node:test';
import { ReplayTimeline } from '../../src/generals_arena/web/replay.js';

function frames(count) {
  const initial={width:2,height:1,tick:0,owner:[[0,1]],army:[[1,2]],structure:[[2,2]],
    terrain:[[0,0]],alive:[true,true],stats:{army:[1,2]},forfeited:[]};
  return [initial,...Array.from({length:count},(_,i)=>({tick:i+1,
    cells:[[i%2,i%2,i+10,i%3]],alive:[true,i<200],
    stats:{army:[i,i+1]},forfeited:i<200?[]:[1]}))];
}
function reference(data,target) {
  const state=structuredClone(data[0]);
  for(const frame of data.slice(1,target+1)) {
    for(const [cell,owner,army,structure] of frame.cells) {
      state.owner[0][cell]=owner;state.army[0][cell]=army;state.structure[0][cell]=structure;
    }
    for(const key of ['tick','alive','stats','forfeited'])state[key]=frame[key];
  }
  return state;
}
test('snapshot seeks match every original tick, including reverse seeks, deaths and cache eviction',()=>{
  const data=frames(700), original=structuredClone(data), replay=new ReplayTimeline(data,{interval:16,capacity:5});
  for(const tick of [700,0,600,201,200,1,699,640,645,400,401,399,0]) {
    assert.deepEqual(replay.seek(tick),reference(data,tick));
    assert.ok(replay.snapshots.size<=5);
  }
  for(let tick=0;tick<=700;tick++)assert.deepEqual(replay.seek(tick),reference(data,tick));
  assert.deepEqual(data,original);
});
test('nearby backward seeks use a snapshot instead of replaying the whole match',()=>{
  const data=frames(1000), replay=new ReplayTimeline(data);
  replay.seek(1000);
  Object.defineProperty(data[1],'cells',{get(){throw Error('replayed from start');}});
  assert.equal(replay.seek(900).tick,900);
  assert.equal(replay.seek(950).tick,950);
});
