import assert from 'node:assert/strict';
import { test } from 'node:test';
import { renderMatrix } from '../../src/generals_arena/web/matrix.js';
import { renderQueue } from '../../src/generals_arena/web/queue.js';

class Element {
  constructor() {this.children=[];this.style={};this.attributes={};this.className='';this.listeners={};}
  append(...children) {this.children.push(...children);}
  replaceChildren(...children) {this.children=children;}
  setAttribute(k,v) {this.attributes[k]=v;}
  addEventListener(key,callback) {this.listeners[key]=callback;}
  querySelector(selector) {
    for (const child of this.children) {
      if (child.className.split(' ').includes(selector.slice(1))) return child;
      const found = child.querySelector(selector); if (found) return found;
    }
    return null;
  }
}
function setup(t) {
  const before = globalThis.document;
  globalThis.document = {createElement:()=>new Element()};
  t.after(()=>{if (before) globalThis.document=before; else delete globalThis.document;});
  return (_id,name,cls)=>Object.assign(new Element(),{textContent:name,className:cls});
}

test('matrix reorders both axes by series/version without moving outcomes to the wrong pair',t=>{
  const playerName = setup(t),host = new Element();
  const cell = win_rate=>({win_rate,draw_rate:0,games:10});
  renderMatrix(host,{entrant_ids:['a10','z','a2','wait','base'],cells:[
    [null,cell(.3),cell(.1),cell(1),cell(.8)],
    [cell(.7),null,cell(.4),cell(1),cell(.8)],
    [cell(.9),cell(.6),null,cell(1),cell(.8)],
    [cell(0),cell(0),cell(0),null,cell(0)],
    [cell(.2),cell(.2),cell(.2),cell(1),null],
  ]},[{entrant_id:'a10',name:'alpha-v10',kind:'cpp',registration_order:10},
      {entrant_id:'z',name:'zeta-v1',kind:'cpp',registration_order:1},
      {entrant_id:'a2',name:'alpha-v2',kind:'cpp',registration_order:2},
      {entrant_id:'wait',name:'WAIT',kind:'cpp',registration_order:3},
      {entrant_id:'base',name:'random',kind:'baseline',registration_order:0}],playerName);
  const [head,body] = host.children[0].children[0].children;
  assert.deepEqual(head.children[0].children.slice(1).map(c=>c.children[0].textContent),
    ['alpha-v10','alpha-v2','zeta-v1','random','WAIT']);
  assert.deepEqual(body.children.map(r=>r.children[0].children[0].textContent),
    ['alpha-v10','alpha-v2','zeta-v1','random','WAIT']);
  assert.deepEqual(body.children[0].children.slice(1).map(c=>c.textContent),['—','10%','30%','80%','100%']);
  assert.deepEqual(body.children[1].children.slice(1).map(c=>c.textContent),['90%','—','60%','80%','100%']);
});

test('queue shows mode progress, separates failures and retains expanded completed history',t=>{
  const playerName=setup(t),host=new Element(),badge=new Element();
  const jobs=[{id:'done',entrant_id:'d',name:'finished',status:'complete',created:1,done:1000,budget:1000,
      modes:{'1v1':{done:500,budget:500},ffa:{done:500,budget:500}}},
    {id:'work',entrant_id:'w',name:'in progress',status:'queued',created:2,done:20,budget:1000,
      modes:{'1v1':{done:12,budget:500},ffa:{done:8,budget:500}}}];
  renderQueue(host,badge,jobs,null,playerName);
  assert.equal(badge.textContent,'1 待完成');
  const card=host.querySelector('.queue-job');
  assert.equal(card.children[1].children[1].value,12);
  assert.equal(card.children[2].children[1].value,8);
  assert.equal(card.children[1].children[1].max,500);
  assert.equal(host.querySelector('.queue-completed').open,false);
  host.querySelector('.queue-completed').open=true;
  jobs[1].status='failed'; jobs[1].error='Policy error';
  renderQueue(host,badge,jobs,'Worker stopped',playerName);
  assert.equal(badge.textContent,'1 项异常');
  assert.equal(host.querySelector('.queue-completed').open,true);
  assert.equal(host.querySelector('.queue-error').textContent,'Policy error');
  assert.equal(host.children.at(-1).textContent,'Worker stopped');
  jobs[1].status='complete';jobs[1].done=1000;
  renderQueue(host,badge,jobs,null,playerName);
  assert.equal(badge.textContent,'全部完成');
  assert.equal(host.querySelector('.queue-job'),null);
  assert.equal(host.querySelector('.queue-completed-list').children.length,2);
});
