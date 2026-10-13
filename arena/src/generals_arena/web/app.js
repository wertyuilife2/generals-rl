import { Battlefield } from './board.js';
import { Playback } from './playback.js';
import { MoveInput } from './input.js';
const $ = id => document.getElementById(id);
const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text !== undefined) n.textContent = text; return n; };
let meta, names = new Map(), pool = 0, page = 0, tab = 'ranking', player = null;
let historyRequest = 0;
let token = '', tokenKey = null;
let live = null, board = null, sending = false;
const outbox = [];
let replay = null, frame = 0, dialogMode = null, polling = false;
const formatter = new Intl.NumberFormat('zh-CN');
const number = n => formatter.format(n);
function notice(message = '') { $('notice').hidden = !message; $('notice').textContent = message; }
async function api(path, body, auth = true) {
  const headers = {};
  if (token && auth) headers.Authorization = `Bearer ${token}`;
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  const response = await fetch(path, {method: body === undefined ? 'GET' : 'POST', headers, body: body === undefined ? undefined : JSON.stringify(body)});
  const data = await response.json();
  if (!response.ok) { const error=new Error(data.error || `HTTP ${response.status}`);error.status=response.status;throw error; }
  return data;
}
function showTab(next) {
  tab = next;
  for (const button of document.querySelectorAll('.tab')) button.classList.toggle('active', button.dataset.tab === next);
  for (const key of ['ranking','history','play']) $(`${key}-panel`).hidden = key !== next;
  if (next === 'history') loadHistory().catch(e => notice(e.message));
  if (next === 'play') renderPlayer();
}
function renderMeta(value) {
  meta = value;
  const nextTokenKey = `generals-arena-token:${meta.league_id || meta.rules_id}`;
  if (tokenKey !== nextTokenKey) {
    tokenKey = nextTokenKey;
    token = localStorage.getItem(tokenKey) || '';
    player = null; live = null;
  }
  names = new Map(meta.entrants.map(e => [e.entrant_id, e.name]));
  $('rating-reference').textContent = names.get(meta.anchor) || 'WAIT';
  const options = ['0', ...meta.config.player_counts.map(String)];
  if ([...$('pool').options].map(n => n.value).join() !== options.join()) {
    for (const id of ['pool', 'play-players']) {
      $(id).replaceChildren(...options.map(p => {
        const option = el('option', '', p==='0' ? (id==='pool'?'全部人数':'随机人数（2–8）') : `${p} 人`);
        option.value=p;
        return option;
      }));
    }
    $('pool').value=pool;
  }
  for (const [id, label] of [['history-entrant', '全部选手'], ['history-opponent', '不限对手']]) {
    const select = $(id), choices = [['', label], ...meta.entrants.map(e => [e.entrant_id, e.name])];
    if (JSON.stringify([...select.options].map(o => [o.value, o.textContent])) !== JSON.stringify(choices)) {
      const selected = select.value;
      select.replaceChildren(...choices.map(([value, name]) => {
        const option = el('option', '', name); option.value = value; return option;
      }));
      select.value = choices.some(([value]) => value === selected) ? selected : '';
    }
  }
  const counts=meta.config.player_counts;
  const first=meta.map_ranges[counts[0]],last=meta.map_ranges[counts[counts.length-1]];
  $('map-description').textContent = `宽高按人数随机：${counts[0]} 人 ${first[0]}–${first[1]} 格，${counts[counts.length-1]} 人 ${last[0]}–${last[1]} 格。同一 AI 可有多个独立实例。`;
  $('pool-caption').textContent = '统一 rating · 所有人数的对局共同计分';
  $('turn-description').textContent = `每 ${meta.config.tick_seconds} 秒执行至多一步。关闭页面后对局继续，按最终结果计分。`;
  const active = meta.jobs.filter(j => j.status !== 'complete');
  $('job-count').textContent = active.length;
  $('jobs').replaceChildren();
  for (const job of [...active, ...meta.jobs.filter(j => j.status === 'complete')].slice(0, 8)) {
    const row = el('div', 'job'), title = el('div', 'job-title');
    title.append(el('span', '', job.name), el('span', '', `${job.done} / ${job.budget}`));
    const progress = el('progress'); progress.max = job.budget; progress.value = job.done;
    row.append(title, progress, el('div', 'job-state', {queued:'等待 / 评测中',failed:'评测失败，可用 CLI 重试',complete:'已完成'}[job.status] || job.status));
    if (job.error) row.append(el('div', 'job-error', job.error));
    $('jobs').append(row);
  }
  if (!meta.jobs.length) $('jobs').append(el('p', 'empty', '暂无评测任务。通过 CLI 登记算法后会自动加入队列。'));
  if (meta.worker_error) $('jobs').append(el('p', 'job-error', meta.worker_error));
  renderPlayer();
}
async function loadRank() {
  const data = await api('/api/leaderboard');
  $('ranking').replaceChildren();
  data.ranking.forEach((row, i) => {
    const tr = el('tr'), name = el('td'), wrap = el('div', 'rank-player'), detail = el('div');
    detail.append(el('div', 'player-name', row.name), el('div', 'player-sub', [row.kind === 'human' ? '人类玩家' : row.kind === 'baseline' || row.anchor ? 'Baseline' : 'Algorithm', row.disqualified ? '策略违规 · 已停用' : row.anchor ? '零分基准' : row.provisional ? '定级中' : '已定级'].join(' · ')));
    wrap.append(el('span', `rank-no${i < 3 ? ' top' : ''}`, String(i + 1).padStart(2, '0')), detail); name.append(wrap);
    const change = Number.isFinite(row.elo_delta) ? Math.round(row.elo_delta * 10) / 10 : null;
    const delta = el('td', `elo-delta${change > 0 ? ' positive' : change < 0 ? ' negative' : ''}`,
      change === null ? '—' : `${change > 0 ? '+' : ''}${change.toFixed(1)}`);
    delta.title = row.previous_version ? `相对 ${row.previous_version} 的当前 Elo${change === null ? '（双方均完成对局后显示）' : ''}` : '暂无可比较的上一版本';
    const elo = el('td', 'elo'), estimate = el('span', 'elo-value', row.elo.toFixed(1));
    const ci = el('span', 'elo-ci', row.elo_ci ? `±${row.elo_ci.half_width.toFixed(1)}` : '± —');
    ci.title = row.anchor ? '零分基准固定为 0' : row.elo_ci
      ? `95% 估计区间 [${row.elo_ci.lower.toFixed(1)}, ${row.elo_ci.upper.toFixed(1)}] · ${row.ci_blocks} 组独立地图；基于名次模型的近似估计`
      : '95% 估计区间：独立地图或基准校准样本不足';
    ci.setAttribute('aria-label', ci.title);
    elo.append(estimate, ci);
    tr.append(name, elo, delta,
      ...['win_rate', 'survival_index', 'kill_index'].map(key =>
        el('td', 'metric', Number.isFinite(row[key]) ? `${(100 * row[key]).toFixed(1)}%` : '—')));
    $('ranking').append(tr);
    if (player && row.entrant_id === player.entrant_id) $('player-rating').textContent = `统一榜单 · ${row.elo.toFixed(1)} Elo · ${row.games} 局`;
  });
}
async function loadHistory() {
  const request = ++historyRequest;
  const query = new URLSearchParams({offset: page * 20, limit: 20});
  if (pool) query.set('players', pool);
  for (const key of ['entrant', 'opponent']) {
    const value = $(`history-${key}`).value;
    if (value) query.set(key, value);
  }
  const filtered = !!(pool || $('history-entrant').value || $('history-opponent').value);
  $('clear-history-filters').disabled = !filtered;
  const data = await api(`/api/matches?${query}`);
  if (request !== historyRequest) return; // A slower previous filter must not replace the latest results.
  $('history').replaceChildren();
  for (const match of data.matches) {
    const row = el('div', 'match-row'), detail = el('div');
    const copies = new Map();
    for (const id of match.entrant_ids) copies.set(id, (copies.get(id)||0)+1);
    detail.append(el('div', 'match-names', [...copies].map(([id,count]) => `${names.get(id)||id.slice(0,8)}${count>1?` ×${count}`:''}`).join(' · ')));
    const winner = match.winner_seat == null ? null : names.get(match.entrant_ids[match.winner_seat]);
    detail.append(el('div', 'match-winner', match.status === 'failed' ? `策略违规 · 未计分 · ${match.policy_error || '执行失败'}` : match.status === 'aborted' ? '服务中断 · 未计分' : winner ? `${winner}（P${match.winner_seat+1}）获胜` : '达到时限 · 存活者并列'));
    if (match.decision_timing?.length) {
      const calls = match.decision_timing.reduce((n, t) => n + t.calls, 0);
      const maximum = Math.max(...match.decision_timing.map(t => t.max_ns)) / 1e6;
      detail.append(el('div', 'match-meta', `C++ 决策 ${calls} 次 · 最大 ${maximum.toFixed(3)} ms · 上限 10 ms`));
    }
    if(match.ranks) {
      const placements=match.entrant_ids.map((id,seat)=>({id,seat,rank:match.ranks[seat]})).sort((a,b)=>a.rank-b.rank||a.seat-b.seat);
      detail.append(el('div','match-meta',placements.map(p=>`#${p.rank} P${p.seat+1} ${names.get(p.id)||p.id.slice(0,8)}`).join(' · ')));
    }
    if(match.rating_changes)detail.append(el('div','match-winner',Object.entries(match.rating_changes).map(([id,c])=>`${names.get(id)||id.slice(0,8)} ${c.delta>=0?'+':''}${c.delta.toFixed(1)}`).join(' · ')));
    detail.append(el('div', 'match-meta', `${match.map.players} 人 · ${match.map.width} × ${match.map.height} · ${match.ticks ?? '—'} ticks · ${new Date(match.finished * 1000).toLocaleString('zh-CN')} · ${match.game_id.slice(0, 8)}`));
    const button = el('button', 'subtle', '回放 ↗'); button.disabled = !['complete', 'failed'].includes(match.status);
    button.addEventListener('click', () => openReplay(match.game_id).catch(e => notice(e.message)));
    row.append(detail, button); $('history').append(row);
  }
  if (!data.matches.length) $('history').append(el('p', 'empty', filtered ? '没有符合筛选条件的对局。' : '还没有对局记录。登记算法或开始你的第一局。'));
  $('page-info').textContent = `第 ${page + 1} 页 · 共 ${data.total} 局`;
  $('prev-page').disabled = page === 0; $('next-page').disabled = (page + 1) * 20 >= data.total;
}
function renderPlayer() {
  $('signup').hidden = !!player; $('player-lobby').hidden = !player;
  if (player) { $('player-label').textContent = player.name; $('play-title').textContent = '准备好下一场对局'; }
  if (meta) {
    const bots = meta.entrants.filter(e => e.kind !== 'human').length;
    $('start-game').disabled = bots === 0;
    const players = Number($('play-players').value);
    $('start-game').textContent = live && !live.done ? '返回当前对局' : bots === 0 ? '需要一个算法对手' : players ? `开始 ${players} 人排位` : '随机匹配并开始';
  }
}
async function restorePlayer() {
  if (!token) return;
  try { player = await api('/api/me'); live = await api('/api/play/active'); localStorage.setItem(tokenKey, token); }
  catch (e) { player = null; token = ''; throw e; }
  renderPlayer();
}
function showDialog(mode) {
  pauseReplay(); dialogMode = mode;
  $('board-dialog').classList.toggle('human-game', mode === 'live');
  $('board-error').textContent = '';
  $('live-controls').hidden = mode !== 'live'; $('replay-controls').hidden = mode !== 'replay';
  $('board-mode').textContent = mode === 'live' ? 'LIVE / RANKED' : 'MATCH ARCHIVE';
  if (!$('board-dialog').open) $('board-dialog').showModal();
  battlefield.reset();
  $('board').focus();
}
function draw() {
  if (!board) return;
  battlefield.render(board, dialogMode==='live'?input.selected:-1, dialogMode==='live'?{half:input.half,queue:input.queue}:{});
  if(dialogMode==='live') { $('half').checked=input.half;$('action-status').textContent=`已排队 ${input.queue.length} 步`; }
  $('turn-counter').textContent=board.tick;
}
let playerNodes=[], playerNames='';
function renderGamePlayers(entries) {
  const signature=entries.map(e=>e.name).join('\0')+board.player_id;
  if(signature!==playerNames) {
    playerNames=signature;
    playerNodes=entries.map((entry,seat)=>{
      const row=el('div','game-player'),name=el('strong'),stats=el('small');
      name.append(el('span',`seat-dot seat-${seat}`),document.createTextNode(`P${seat+1} · ${entry.name}${board.player_id===seat?' · 你':''}`));
      row.append(name,stats);return {row,stats};
    });
    $('game-players').replaceChildren(...playerNodes.map(p=>p.row));
  }
  const order=entries.map((_,seat)=>seat).sort((a,b)=>board.stats.army[b]-board.stats.army[a] || board.stats.territory[b]-board.stats.territory[a] || a-b);
  for(const [place,seat] of order.entries()) {
    const {row,stats}=playerNodes[seat],text=`${board.forfeited?.includes(seat)?'超时投降 · ':board.alive[seat]?'':'已淘汰 · '}兵力 ${number(board.stats.army[seat])} · 领地 ${number(board.stats.territory[seat])}`;
    if(stats.textContent!==text)stats.textContent=text;
    row.classList.toggle('eliminated',!board.alive[seat]);
    row.classList.toggle('forfeited',!!board.forfeited?.includes(seat));
    const current=$('game-players').children[place];
    if(current!==row)$('game-players').insertBefore(row,current||null);
  }
}
function updateLive(value, force=false) {
  if(!force&&value.revision<=input.revision)return;
  if(!input.observe(value))return;
  live=value;
  if(dialogMode!=='live'||!$('board-dialog').open)return;
  board=live.board;
  $('board-title').textContent=`${live.entrants.length} 人排位 · ${board.width} × ${board.height}`;
  $('board-status').textContent=live.error?`本局未计分：${live.error}`:live.done?`对局结束 · ${live.result.winner_seat==null?'达到时限，存活者并列':`${live.entrants[live.result.winner_seat].name} 获胜`} · 已计入排行榜`:`Tick ${board.tick} / ${live.max_ticks} · ${board.alive[board.player_id]?'局部视野 · 输入立即排队，每 tick 执行一步':'你已被淘汰，等待对局结束后计分'}`;
  draw();renderGamePlayers(live.entrants);
  if(live.done)renderPlayer();
}
async function openLive() {
  notice();
  if (!player) { showTab('play'); return; }
  $('start-game').disabled = true;
  try {
    const value = await api('/api/play', {players: Number($('play-players').value)||null});
    const fresh=live?.game_id!==value.game_id || !input.board;
    if(fresh)input.reset();
    live=value;showDialog('live');updateLive(value,true);
    if(fresh)input.home();
    poll();
  } finally { renderPlayer(); }
}
const input = new MoveInput(()=>draw(),command=>{
  outbox.push({gameId:live.game_id,command});flushCommands();
});
async function flushCommands() {
  if(sending)return;
  sending=true;
  try {
    while(outbox.length) {
      const {gameId,command}=outbox[0];
      try {
        const value=await api(`/api/play/${gameId}/commands`,command);
        outbox.shift();
        if(live?.game_id===gameId)updateLive(value);
        $('board-error').textContent='';
      } catch(error) {
        $('board-error').textContent=error.status?error.message:'连接中断，正在重试操作…';
        if(!error.status){setTimeout(flushCommands,300);break;}
        outbox.length=0;input.pending=[];
        const value=await api(`/api/play/${gameId}`);
        input.next=value.last_command_id+1;updateLive(value,true);break;
      }
    }
  } catch(error) { $('board-error').textContent=error.message; }
  finally { sending=false; }
}
function move(direction,half=false,wait=false) {
  if(dialogMode!=='live'||!live||live.done)return;
  if(wait)input.issue('append',0);else input.move(direction,half);
}
async function openReplay(id) {
  notice('正在校验并读取回放…');
  const data = await api(`/api/replay/${id}`);
  replay = data; frame = 0;
  showDialog('replay'); $('timeline').max = data.frames.length - 1;
  $('board-title').textContent = `${data.result.map.players} 人对局回放 · ${id.slice(0,8)}`;
  $('board-status').textContent = `${data.result.map.width} × ${data.result.map.height} · ${data.result.policy_error ? '策略违规中断 · 未计分' : data.result.reason === 'surrender' ? '投降结算' : data.result.reason === 'capture' ? '主城占领' : '达到时限'} · 已校验全部已执行动作与状态`;
  seek(0); notice();
}
function applyFrame(next) {
  const f = replay.frames[next];
  for (const [cell,owner,army,structure] of f.cells) { const y=Math.floor(cell/board.width),x=cell%board.width;board.owner[y][x]=owner;board.army[y][x]=army;board.structure[y][x]=structure; }
  board.tick=f.tick;board.alive=f.alive;board.stats=f.stats;board.forfeited=f.forfeited;
}
function seek(next) {
  next=Math.max(0,Math.min(replay.frames.length-1,next));
  if (!board || next < frame || next === 0 || frame === 0) {board=structuredClone(replay.frames[0]);frame=0;}
  for (let i=frame+1;i<=next;i++) applyFrame(i);
  frame=next;$('timeline').value=frame;$('replay-tick').textContent=`Tick ${board.tick} / ${replay.result.ticks}`;
  draw();renderGamePlayers(replay.result.entrant_ids.map(id=>({name:names.get(id)||id.slice(0,8)})));
}
const playback = new Playback({
  interval: () => (meta?.config.tick_seconds || .5) * 1000 / Number($('replay-speed').value),
  step: () => { seek(frame + 1); return frame < replay.frames.length - 1; },
  stopped: () => { $('toggle-replay').textContent = '播放'; },
});
function pauseReplay(){playback.stop();}
$('toggle-replay').addEventListener('click',()=>{
  if(playback.running){pauseReplay();return;}if(frame===replay.frames.length-1)seek(0);
  $('toggle-replay').textContent='暂停';playback.start();
});
$('timeline').addEventListener('input',()=>{pauseReplay();seek(Number($('timeline').value));});
$('close-board').addEventListener('click',()=>{$('board-dialog').close();});
$('board-dialog').addEventListener('close',()=>{pauseReplay();dialogMode=null;refresh().catch(e=>notice(e.message));});
const battlefield = new Battlefield($('board'),(x,y,half)=>{
  if(dialogMode==='live'&&live&&!live.done)input.select(y*board.width+x,half);
});
$('zoom-in').addEventListener('click',()=>battlefield.adjustZoom(1.2));
$('zoom-out').addEventListener('click',()=>battlefield.adjustZoom(1/1.2));
$('zoom-reset').addEventListener('click',()=>battlefield.reset());
$('half').addEventListener('change',()=>{input.half=$('half').checked;draw();});
$('clear-moves').addEventListener('click',()=>input.clear());
$('undo-move').addEventListener('click',()=>input.undo());
$('board-dialog').addEventListener('keydown',event=>{
  if(dialogMode!=='live'||!live||live.done||event.ctrlKey||event.metaKey||event.altKey)return;
  const key=event.key.toLowerCase();
  const direction={w:0,arrowup:0,d:1,arrowright:1,s:2,arrowdown:2,a:3,arrowleft:3}[key];
  if(direction!==undefined){event.preventDefault();move(direction,event.shiftKey);}
  else if(key===' '){event.preventDefault();input.deselect();}
  else if(key==='q'){event.preventDefault();input.clear();}
  else if(key==='e'){event.preventDefault();input.undo();}
  else if(key==='h'||key==='g'){event.preventDefault();input.home();}
  else if(key==='z'){event.preventDefault();input.half=!input.half;draw();}
});
for(const button of document.querySelectorAll('[data-dir]'))button.addEventListener('click',()=>move(Number(button.dataset.dir)));
for(const button of document.querySelectorAll('.tab'))button.addEventListener('click',()=>showTab(button.dataset.tab));
function filterHistory() {
  pool = Number($('pool').value); page = 0;
  loadHistory().catch(e => notice(e.message));
}
for (const id of ['pool', 'history-entrant', 'history-opponent']) $(id).addEventListener('change', filterHistory);
$('history-filter').addEventListener('submit', event => { event.preventDefault(); filterHistory(); });
$('clear-history-filters').addEventListener('click', () => {
  $('pool').value = '0'; $('history-entrant').value = ''; $('history-opponent').value = '';
  filterHistory();
});
$('prev-page').addEventListener('click',()=>{page=Math.max(0,page-1);loadHistory().catch(e=>notice(e.message));});
$('next-page').addEventListener('click',()=>{page++;loadHistory().catch(e=>notice(e.message));});
$('signup-form').addEventListener('submit',async event=>{event.preventDefault();try{const data=await api('/api/players',{name:$('player-name').value},false);token=data.token;player=data.entrant;localStorage.setItem(tokenKey,token);notice();await refresh();renderPlayer();}catch(e){notice(e.message);}});
$('recover-form').addEventListener('submit',async event=>{event.preventDefault();token=$('recover-token').value.trim();try{await restorePlayer();localStorage.setItem(tokenKey,token);$('recover-token').value='';notice('玩家身份已恢复');await refresh();}catch(e){notice(e.message);}});
$('copy-token').addEventListener('click',async()=>{try{await navigator.clipboard.writeText(token);notice('身份凭证已复制。请保存在你自己的密码管理器中。');}catch(e){notice('浏览器未允许复制凭证。请允许剪贴板权限后重试。');}});
$('start-game').addEventListener('click',()=>openLive().catch(e=>notice(e.message)));
async function refresh(){renderMeta(await api('/api/meta'));if(player&&!$('board-dialog').open){live=await api('/api/play/active');renderPlayer();}await loadRank();if(tab==='history')await loadHistory();$('connection').textContent='联赛在线';}
async function poll(){
  if(polling)return;
  if(!(player&&dialogMode==='live'&&live&&!live.done&&$('board-dialog').open))return;
  polling=true;
  const id=live.game_id;
  try{const value=await api(`/api/play/${id}?after=${Math.max(0,input.revision)}`);if(live?.game_id===id)updateLive(value);}
  catch(e){$('board-error').textContent=e.message;}
  finally{polling=false;setTimeout(poll,50);}
}
(async()=>{
  try{await refresh();}catch(e){notice(e.message);$('connection').textContent='连接异常';return;}
  if(token){
    try{await restorePlayer();await loadRank();}
    catch(e){notice(e.status===403?'保存的玩家凭证无法用于此联赛，请恢复身份或创建玩家。':e.message);renderPlayer();}
  }
})();
setInterval(poll,1000);
setInterval(()=>{if(!$('board-dialog').open)refresh().catch(e=>{$('connection').textContent='连接中断';notice(e.message);});},5000);
