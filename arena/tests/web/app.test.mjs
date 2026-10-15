import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { test } from 'node:test';

// Run the actual application entry point with a small DOM and HTTP fixture.
// Renderer tests cover canvas behavior separately; this catches startup and
// refresh failures that isolated component tests cannot see.
class Element {
  constructor() {
    this.children = []; this.dataset = {}; this.attributes = {}; this.listeners = {};
    this.style = {setProperty() {}}; this.classList = {toggle() {}};
    this.textContent = ''; this.value = ''; this.open = false;
  }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  get options() { return this.children; }
  get firstElementChild() { return this.children[0]; }
  setAttribute(key, value) { this.attributes[key] = value; }
  getAttribute(key) { return this.attributes[key]; }
  addEventListener(key, callback) { this.listeners[key] = callback; }
  focus() {}
  showModal() { this.open = true; }
  close() { this.open = false; this.listeners.close?.(); }
}

test('application boots, switches rating pools and refreshes without scope errors', async t => {
  const html = await readFile(new URL('../../src/generals_arena/web/index.html', import.meta.url), 'utf8');
  const ids = new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(([, id]) => [id, new Element()]));
  const buttons = ['1v1', 'ffa'].map(mode => Object.assign(new Element(), {dataset:{rankingMode:mode}}));
  const intervals = new Map(), requests = [];
  const metadata = {league_id:'test', anchor:'wait', config:{player_counts:[2,3], tick_seconds:.5},
    map_ranges:{2:[18,23],3:[19,25]}, jobs:[], entrants:[{entrant_id:'wait', name:'WAIT', kind:'cpp'}]};
  const globals = {
    document:{getElementById:id => ids.get(id), createElement:() => new Element(),
      querySelectorAll:selector => selector === '[data-ranking-mode]' ? buttons : []},
    localStorage:{getItem:() => null}, Image:class {}, ResizeObserver:class {observe() {}},
    setInterval:(callback, delay) => intervals.set(delay, callback),
    fetch:async path => {
      requests.push(path);
      if (path === '/api/meta') return {ok:true, json:async () => metadata};
      assert.match(path, /^\/api\/leaderboard\?mode=(1v1|ffa)$/);
      const duel = path.endsWith('1v1');
      return {ok:true, json:async () => ({anchor:duel?'wait':null, completed_games:duel?12:8,
        win_matrix:{entrant_ids:['wait'],cells:[[null]]},
        title_bands:duel?[{key:'newbie',label:'Newbie',color:'#aaa',minimum:null,maximum:1200}]:[],
        ranking:[{entrant_id:'wait', name:'WAIT', kind:'cpp', anchor:duel, elo:duel?0:1000,
          provisional:!duel, games:duel?12:0,
          rating_title:duel?{key:'newbie',label:'Newbie',color:'#aaa',minimum:null,maximum:1200}:null,
          duel_games:12,duel_wins:3,duel_win_rate:.25,draw_rate:.5,mean_win_ticks:120,mean_decision_ms:.12,timed_games:12,
          win_rate:.3,survival_index:.6,kill_index:.2}]})};
    },
  };
  const saved = new Map(Object.keys(globals).map(key => [key, Object.getOwnPropertyDescriptor(globalThis,key)]));
  t.after(() => { for (const [key, descriptor] of saved) {
    if (descriptor) Object.defineProperty(globalThis,key,descriptor); else delete globalThis[key];
  }});
  Object.assign(globalThis, globals);
  const settle = () => new Promise(resolve => setImmediate(resolve));
  await import('../../src/generals_arena/web/app.js');
  await settle();
  assert.equal(ids.get('connection').textContent, '联赛在线', ids.get('notice').textContent);
  assert.equal(ids.get('ranking').children.length, 1);
  assert.equal(ids.get('win-matrix').children.length, 1);
  assert.equal(ids.get('rating-reference').textContent, 'WAIT = 0');
  assert.match(ids.get('pool-caption').textContent, /1v1.*12 局.*GoR/);
  assert.deepEqual(ids.get('ranking-head').children.slice(3).map(c=>c.textContent),['胜率','和棋率','获胜步数','决策耗时']);
  assert.deepEqual(ids.get('ranking').children[0].children.slice(3).map(c=>c.textContent),['25.0%','50.0%','120 步','120.0 μs']);
  assert.deepEqual(ids.get('ranking').children[0].children[0].children[0].children[1].children[0].children.map(c=>c.textContent),['WAIT','Newbie']);
  buttons[1].listeners.click(); await settle();
  assert.equal(ids.get('rating-reference').textContent, '均值 1000');
  assert.match(ids.get('pool-caption').textContent, /FFA.*8 局.*Codeforces/);
  assert.match(ids.get('matrix-caption').textContent, /最终名次/);
  assert.deepEqual(ids.get('ranking-head').children.slice(3).map(c=>c.textContent),['胜率','存活指数','击杀指数']);
  const unchangedRow=ids.get('ranking').children[0];
  intervals.get(5000)(); await settle();
  assert.equal(ids.get('ranking').children[0],unchangedRow);
  assert.equal(ids.get('connection').textContent, '联赛在线', ids.get('notice').textContent);
  assert.equal(ids.get('rating-reference').textContent, '均值 1000');
  assert.match(ids.get('pool-caption').textContent, /FFA.*8 局.*Codeforces/);
  buttons[0].listeners.click(); await settle();
  assert.equal(ids.get('rating-reference').textContent, 'WAIT = 0');
  assert.equal(ids.get('notice').textContent, '');
  assert.deepEqual(requests, ['/api/meta','/api/leaderboard?mode=1v1','/api/leaderboard?mode=ffa',
    '/api/meta','/api/leaderboard?mode=ffa','/api/leaderboard?mode=1v1']);
});

test('lobby migrates its existing account, adds and switches accounts, and submits selected opponents', async t=>{
  const html=await readFile(new URL('../../src/generals_arena/web/index.html',import.meta.url),'utf8');
  const ids=new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(([,id])=>[id,new Element()]));
  const accounts={a:{entrant_id:'a',name:'Alice',kind:'human'},b:{entrant_id:'b',name:'Bob',kind:'human'}};
  const values=new Map([['generals-arena-token:accounts-test','token-a']]),requests=[];
  const meta={league_id:'accounts-test',anchor:'wait',config:{player_counts:[2,4],tick_seconds:.5},
    map_ranges:{2:[18,23],4:[20,26]},jobs:[],entrants:[
      {entrant_id:'wait',name:'WAIT',kind:'cpp'}, {entrant_id:'bot',name:'bot-v1',kind:'cpp'},
      {entrant_id:'disabled',name:'disabled-v1',kind:'cpp',disqualified:true},accounts.a]};
  const globals={
    document:{getElementById:id=>ids.get(id),createElement:()=>new Element(),querySelectorAll:()=>[]},
    localStorage:{getItem:key=>values.get(key),setItem:(key,value)=>values.set(key,value)},
    Image:class{},ResizeObserver:class{observe(){}},setInterval:()=>{},
    fetch:async(path,options)=>{
      requests.push({path,options});let data;
      if(path==='/api/meta')data=meta;
      else if(path.startsWith('/api/leaderboard'))data={anchor:'wait',completed_games:0,ranking:[],win_matrix:{entrant_ids:[],cells:[]}};
      else if(path==='/api/players'){meta.entrants.push(accounts.b);data={token:'token-b',entrant:accounts.b};}
      else if(path==='/api/me'){
        const key=options.headers.Authorization?.slice(-1);
        if(!accounts[key])return {ok:false,status:403,json:async()=>({error:'invalid credential'})};
        data=accounts[key];
      }
      else if(path==='/api/play/active')data=null;
      // Stop before canvas initialization; the HTTP/game tests cover execution.
      else if(path==='/api/play')return {ok:false,status:409,json:async()=>({error:'fixture intercepted game'})};
      else throw Error(path);
      return {ok:true,json:async()=>data};
    },
  };
  const saved=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
  t.after(()=>{for(const [key,descriptor] of saved) {
    if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];
  }});
  Object.assign(globalThis,globals);
  const settle=()=>new Promise(resolve=>setImmediate(resolve)), event={preventDefault(){}};
  await import('../../src/generals_arena/web/app.js?accounts-test');await settle();
  assert.equal(ids.get('player-label').textContent,'Alice');
  assert.equal(ids.get('play-account').options.length,1);
  ids.get('new-account').listeners.click();
  assert.equal(ids.get('signup').hidden,false);
  ids.get('player-name').value='Bob';
  await ids.get('signup-form').listeners.submit(event);
  assert.equal(ids.get('play-account').options.length,2);
  assert.equal(ids.get('player-label').textContent,'Bob');
  ids.get('play-account').value='a';await ids.get('play-account').listeners.change();
  assert.equal(ids.get('player-label').textContent,'Alice');
  ids.get('recover-token').value='invalid';await ids.get('recover-form').listeners.submit(event);
  assert.equal(ids.get('player-label').textContent,'Alice');
  assert.equal(values.get('generals-arena-token:accounts-test'),'token-a');
  ids.get('play-players').value='4';ids.get('play-players').listeners.change();
  const selectors=ids.get('play-opponents').children.map(label=>label.children[1]);
  assert.equal(selectors.length,3);
  assert.deepEqual(new Set(selectors[0].options.map(option=>option.value)),new Set(['','bot','wait']));
  for(const select of selectors.slice(0,2)){select.value='bot';select.listeners.change();}
  await ids.get('start-game').listeners.click();
  const request=requests.findLast(r=>r.path==='/api/play');
  assert.equal(request.options.headers.Authorization,'Bearer token-a');
  assert.deepEqual(JSON.parse(request.options.body),{players:4,opponents:['bot','bot',null]});
  assert.equal(JSON.parse(values.get('generals-arena-accounts:accounts-test')).accounts.length,2);
});

test('replay opens immediately, cancels closed loads and ignores stale responses', async t => {
  const html = await readFile(new URL('../../src/generals_arena/web/index.html', import.meta.url), 'utf8');
  const ids = new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(([,id]) => [id,new Element()]));
  const tab = Object.assign(new Element(), {dataset:{tab:'history'}}), pending = [];
  const metadata = {league_id:'replay-test', config:{player_counts:[2],tick_seconds:.5},
    map_ranges:{2:[18,23]}, jobs:[], entrants:[]};
  const board = {tick:0,width:1,height:1,player_id:-1,owner:[[-1]],army:[[0]],structure:[[0]],
    terrain:[[0]],alive:[],stats:{army:[],territory:[],cities:[]}};
  const payload = {result:{ticks:0,map:{players:2,width:1,height:1},entrant_ids:[]},frames:[board]};
  const response = data => ({ok:true,json:async()=>data});
  const globals = {
    document:{getElementById:id=>ids.get(id),createElement:()=>new Element(),
      querySelectorAll:selector=>selector==='.tab'?[tab]:[]},
    localStorage:{getItem:()=>null}, Image:class{}, ResizeObserver:class{observe(){}},setInterval:()=>{},
    fetch:(path, options)=>{
      if(path==='/api/meta')return Promise.resolve(response(metadata));
      if(path.startsWith('/api/leaderboard'))return Promise.resolve(response({completed_games:0,ranking:[],win_matrix:{entrant_ids:[],cells:[]}}));
      if(path.startsWith('/api/matches'))return Promise.resolve(response({total:1,matches:[
        {game_id:'replay-one',status:'complete',finished:1,ticks:0,entrant_ids:[]}]}));
      assert.equal(path,'/api/replay/replay-one');
      return new Promise(resolve=>pending.push({resolve,signal:options.signal}));
    },
  };
  const saved = new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
  t.after(()=>{for(const [key,descriptor] of saved) {
    if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];
  }});
  Object.assign(globalThis,globals);
  const settle=()=>new Promise(resolve=>setImmediate(resolve));
  await import('../../src/generals_arena/web/app.js?replay-loading');await settle();
  tab.listeners.click();await settle();
  const row=ids.get('history').children[0], open=row.children[0].listeners.click;
  tab.listeners.click();await settle();
  assert.equal(ids.get('history').children[0],row); // Unchanged polling preserves the row.
  const first=open();
  assert.equal(ids.get('board-dialog').open,true);
  assert.equal(ids.get('replay-loading').hidden,false);
  assert.equal(ids.get('toggle-replay').disabled,true);
  ids.get('close-board').listeners.click();
  assert.equal(pending[0].signal.aborted,true);
  const second=open();
  pending[0].resolve(response(payload));await first;
  assert.equal(ids.get('toggle-replay').disabled,true); // The closed request cannot replace this one.
  pending[1].resolve(response(payload));await second;
  assert.equal(ids.get('replay-loading').hidden,true);
  assert.equal(ids.get('board').hidden,false);
  assert.equal(ids.get('toggle-replay').disabled,false);
  assert.equal(ids.get('replay-tick').textContent,'Tick 0 / 0');
  ids.get('close-board').listeners.click();
  const failure=open();
  pending[2].resolve({ok:false,status:500,json:async()=>({error:'invalid replay'})});await failure;
  assert.match(ids.get('replay-loading').textContent,/加载失败.*invalid replay/);
  assert.equal(ids.get('toggle-replay').disabled,true);
  assert.equal(ids.get('board-dialog').attributes['aria-busy'],'false');
  ids.get('close-board').listeners.click();await settle();
});
