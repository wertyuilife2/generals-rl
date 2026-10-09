'use strict';
const $ = id => document.getElementById(id);
const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text !== undefined) n.textContent = text; return n; };
const colors = ['#ce565a','#5c92d6','#65b279','#d9ad4d','#a384cf','#de8650','#64c2bb','#d087b1'];
let meta, names = new Map(), pool = 2, page = 0, tab = 'ranking', player = null;
let token = localStorage.getItem('generals-arena-token') || '';
let live = null, board = null, selected = -1, pendingMove = null, sending = false;
let replay = null, frame = 0, replayTimer = null, dialogMode = null, polling = false;
const number = n => Number(n).toLocaleString('zh-CN');
function notice(message = '') { $('notice').hidden = !message; $('notice').textContent = message; }
async function api(path, body, auth = true) {
  const headers = {};
  if (token && auth) headers.Authorization = `Bearer ${token}`;
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  const response = await fetch(path, {method: body === undefined ? 'GET' : 'POST', headers, body: body === undefined ? undefined : JSON.stringify(body)});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
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
  names = new Map(meta.entrants.map(e => [e.entrant_id, e.name]));
  $('entrant-count').textContent = number(meta.entrants.length);
  const options = meta.config.player_counts.map(String);
  if ([...$('pool').options].map(n => n.value).join() !== options.join()) {
    $('pool').replaceChildren(...options.map(p => { const option = el('option', '', `${p} 人对战`); option.value = p; return option; }));
    if (!options.includes(String(pool))) pool = Number(options[0]);
    $('pool').value = pool;
  }
  const range = meta.map_ranges[pool];
  $('map-description').textContent = `${pool} 人地图：宽、高各随机 ${range[0]}–${range[1]} 格。山地与城市密度随机，所有可通行区域相连。`;
  $('pool-caption').textContent = `${pool} 人独立排名池 · ${range[0]}–${range[1]} 格随机宽高`;
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
  const data = await api(`/api/leaderboard?players=${pool}`);
  $('game-count').textContent = number(data.completed_games);
  $('ranking').replaceChildren();
  data.ranking.forEach((row, i) => {
    const tr = el('tr'), name = el('td'), wrap = el('div', 'rank-player'), detail = el('div');
    detail.append(el('div', 'player-name', row.name), el('div', 'player-sub', [row.kind === 'human' ? '人类玩家' : row.kind === 'baseline' ? 'Baseline' : 'Algorithm', row.anchor ? '零分基准' : row.provisional ? '定级中' : '已定级'].join(' · ')));
    wrap.append(el('span', `rank-no${i < 3 ? ' top' : ''}`, String(i + 1).padStart(2, '0')), detail); name.append(wrap);
    tr.append(name, el('td', 'elo', row.elo.toFixed(1)), el('td', '', number(row.games)), el('td', '', `${row.wins} / ${row.losses} / ${row.draws}`), el('td', '', row.games ? `${(100 * row.wins / row.games).toFixed(1)}%` : '—'));
    $('ranking').append(tr);
    if (player && row.entrant_id === player.entrant_id) $('player-rating').textContent = `${pool} 人榜 · ${row.elo.toFixed(1)} Elo · ${row.games} 局`;
  });
}
async function loadHistory() {
  const data = await api(`/api/matches?players=${pool}&offset=${page * 20}&limit=20`);
  $('history').replaceChildren();
  for (const match of data.matches) {
    const row = el('div', 'match-row'), detail = el('div');
    detail.append(el('div', 'match-names', match.entrant_ids.map(id => names.get(id) || id.slice(0, 8)).join('  ·  ')));
    const winner = match.winner_seat == null ? null : names.get(match.entrant_ids[match.winner_seat]);
    detail.append(el('div', 'match-winner', match.status === 'aborted' ? '服务中断 · 未计分' : winner ? `${winner} 获胜` : '达到时限 · 存活者并列'));
    detail.append(el('div', 'match-meta', `${match.map.width} × ${match.map.height} · ${match.ticks ?? '—'} ticks · ${new Date(match.finished * 1000).toLocaleString('zh-CN')} · ${match.game_id.slice(0, 8)}`));
    const button = el('button', 'subtle', '回放 ↗'); button.disabled = match.status !== 'complete';
    button.addEventListener('click', () => openReplay(match.game_id).catch(e => notice(e.message)));
    row.append(detail, button); $('history').append(row);
  }
  if (!data.matches.length) $('history').append(el('p', 'empty', '还没有对局记录。登记算法或开始你的第一局。'));
  $('page-info').textContent = `第 ${page + 1} 页 · 共 ${data.total} 局`;
  $('prev-page').disabled = page === 0; $('next-page').disabled = (page + 1) * 20 >= data.total;
}
function renderPlayer() {
  $('signup').hidden = !!player; $('player-lobby').hidden = !player;
  if (player) { $('player-label').textContent = player.name; $('play-title').textContent = '准备好下一场对局'; }
  if (meta) {
    const bots = meta.entrants.filter(e => e.kind !== 'human').length;
    $('start-game').disabled = bots < pool - 1;
    $('start-game').textContent = live && !live.done ? '返回当前对局' : bots < pool - 1 ? `还需要 ${pool - 1 - bots} 个算法对手` : `开始 ${pool} 人排位`;
  }
}
async function restorePlayer() {
  if (!token) return;
  try { player = await api('/api/me'); live = await api('/api/play/active'); }
  catch (e) { player = null; token = ''; localStorage.removeItem('generals-arena-token'); throw e; }
  renderPlayer();
}
function showDialog(mode) {
  pauseReplay(); dialogMode = mode;
  $('board-error').textContent = '';
  $('live-controls').hidden = mode !== 'live'; $('replay-controls').hidden = mode !== 'replay';
  $('board-mode').textContent = mode === 'live' ? 'LIVE / RANKED' : 'MATCH ARCHIVE';
  if (!$('board-dialog').open) $('board-dialog').showModal();
  $('board').focus();
}
function draw() {
  if (!board) return;
  const canvas = $('board'), cell = 32, ratio = window.devicePixelRatio || 1;
  const w = board.width, h = board.height;
  canvas.width = w * cell * ratio; canvas.height = h * cell * ratio;
  const ctx = canvas.getContext('2d'); ctx.scale(ratio, ratio);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const owner = board.owner[y][x], army = board.army[y][x], terrain = board.terrain[y][x], structure = board.structure[y][x], fog = owner === -3;
    ctx.fillStyle = fog ? '#121c1f' : owner >= 0 ? colors[owner] : terrain === 1 ? '#394441' : structure === 1 ? '#696650' : '#c0c7ba';
    ctx.fillRect(x * cell, y * cell, cell - 1, cell - 1);
    if (fog) { ctx.fillStyle = '#273335'; ctx.fillRect(x * cell + 15, y * cell + 15, 2, 2); continue; }
    if (terrain === 1) {
      ctx.fillStyle = '#6c7770'; ctx.beginPath(); ctx.moveTo(x*cell+6,y*cell+25); ctx.lineTo(x*cell+16,y*cell+7); ctx.lineTo(x*cell+26,y*cell+25); ctx.fill(); continue;
    }
    if (structure) {
      ctx.fillStyle = 'rgba(0,0,0,.23)';
      if (structure === 2) {ctx.beginPath(); ctx.moveTo(x*cell+5,y*cell+9);ctx.lineTo(x*cell+10,y*cell+13);ctx.lineTo(x*cell+16,y*cell+5);ctx.lineTo(x*cell+22,y*cell+13);ctx.lineTo(x*cell+27,y*cell+9);ctx.lineTo(x*cell+24,y*cell+26);ctx.lineTo(x*cell+8,y*cell+26);ctx.closePath();ctx.fill();}
      else {ctx.fillRect(x*cell+7,y*cell+7,18,19);ctx.strokeStyle='rgba(255,255,255,.25)';ctx.strokeRect(x*cell+7,y*cell+7,18,19);}
    }
    if (army > 0 || owner >= 0) {
      ctx.fillStyle = owner >= 0 || structure ? '#fff' : '#253329'; ctx.font = `600 ${army >= 1000 ? 9 : 12}px -apple-system,sans-serif`;ctx.textAlign='center';ctx.textBaseline='middle';
      ctx.fillText(army >= 100000 ? `${Math.round(army / 1000)}k` : String(army), x*cell+16,y*cell+18);
    }
    if (dialogMode === 'live' && y*w+x === selected) {ctx.strokeStyle='#f2f8c7';ctx.lineWidth=3;ctx.strokeRect(x*cell+2,y*cell+2,cell-5,cell-5);}
  }
}
function renderGamePlayers(entries) {
  $('game-players').replaceChildren();
  entries.forEach((entry, seat) => {
    const row = el('div', 'game-player'), name = el('strong');
    name.append(el('span', `seat-dot seat-${seat}`), document.createTextNode(`${entry.name}${board.player_id === seat ? ' · 你' : ''}`));
    row.append(name, el('small', '', `${board.alive[seat] ? '' : '已淘汰 · '}兵力 ${number(board.stats.army[seat])} / 领地 ${number(board.stats.territory[seat])}`));
    $('game-players').append(row);
  });
}
function updateLive(value) {
  live = value;
  if (dialogMode !== 'live' || !$('board-dialog').open) return;
  board = live.board;
  if (pendingMove && board.tick > pendingMove.tick) {
    if (board.owner[Math.floor(pendingMove.target / board.width)][pendingMove.target % board.width] === board.player_id) selected = pendingMove.target;
    pendingMove = null;
  }
  if (selected < 0 || selected >= board.width*board.height || board.owner[Math.floor(selected/board.width)][selected%board.width] !== board.player_id) selected = board.general_pos;
  $('board-title').textContent = `${live.entrants.length} 人排位 · ${board.width} × ${board.height}`;
  $('board-status').textContent = live.error ? `本局未计分：${live.error}` : live.done ? `对局结束 · ${live.result.winner_seat == null ? '达到时限，存活者并列' : `${live.entrants[live.result.winner_seat].name} 获胜`} · 已计入排行榜` : `Tick ${board.tick} / ${live.max_ticks} · ${board.alive[board.player_id] ? '局部视野 · 点击领地选择起点' : '你已被淘汰，等待对局结束后计分'}`;
  draw(); renderGamePlayers(live.entrants);
  if (live.done) renderPlayer();
}
async function openLive() {
  notice();
  if (!player) { showTab('play'); return; }
  $('start-game').disabled = true;
  try {
    live = await api('/api/play', {players: pool});
    selected = live.board.general_pos; pendingMove = null;
    showDialog('live'); updateLive(live);
  } finally { renderPlayer(); }
}
async function move(direction, half = false, wait = false) {
  if (sending || dialogMode !== 'live' || !live || live.done || !board.alive[board.player_id]) return;
  if (!wait && selected < 0) return;
  const tick = board.tick, source = selected, y = Math.floor(source / board.width), x = source % board.width;
  const dx = [0,1,0,-1][direction], dy = [-1,0,1,0][direction];
  if (!wait && (x+dx<0 || x+dx>=board.width || y+dy<0 || y+dy>=board.height)) return;
  sending = true;
  try {
    const data = await api(`/api/play/${live.game_id}/action`, {action: wait ? 0 : 1 + source * 8 + direction * 2 + Number(half || $('half').checked), tick});
    pendingMove = wait ? null : {tick, target: (y+dy)*board.width+x+dx};
    $('action-status').textContent = `已提交 Tick ${data.queued_for_tick}`;
  } catch(e) { $('action-status').textContent = e.message; }
  finally { sending = false; }
}
async function openReplay(id) {
  notice('正在校验并读取回放…');
  const data = await api(`/api/replay/${id}`);
  replay = data; frame = 0;
  showDialog('replay'); $('timeline').max = data.frames.length - 1;
  $('board-title').textContent = `${data.result.map.players} 人对局回放 · ${id.slice(0,8)}`;
  $('board-status').textContent = `${data.result.map.width} × ${data.result.map.height} · ${data.result.reason === 'capture' ? '主城占领' : '达到时限'} · 已校验全部动作与状态`;
  seek(0); notice();
}
function applyFrame(next) {
  const f = replay.frames[next];
  for (const [cell,owner,army,structure] of f.cells) { const y=Math.floor(cell/board.width),x=cell%board.width;board.owner[y][x]=owner;board.army[y][x]=army;board.structure[y][x]=structure; }
  board.tick=f.tick;board.alive=f.alive;board.stats=f.stats;
}
function seek(next) {
  next=Math.max(0,Math.min(replay.frames.length-1,next));
  if (!board || next < frame || next === 0 || frame === 0) {board=structuredClone(replay.frames[0]);frame=0;}
  for (let i=frame+1;i<=next;i++) applyFrame(i);
  frame=next;$('timeline').value=frame;$('replay-tick').textContent=`Tick ${board.tick} / ${replay.result.ticks}`;
  draw();renderGamePlayers(replay.result.entrant_ids.map(id=>({name:names.get(id)||id.slice(0,8)})));
}
function pauseReplay(){if(replayTimer)clearInterval(replayTimer);replayTimer=null;$('toggle-replay').textContent='播放';}
$('toggle-replay').addEventListener('click',()=>{
  if(replayTimer){pauseReplay();return;}if(frame===replay.frames.length-1)seek(0);
  $('toggle-replay').textContent='暂停';replayTimer=setInterval(()=>{seek(frame+Number($('replay-speed').value));if(frame===replay.frames.length-1)pauseReplay();},250);
});
$('timeline').addEventListener('input',()=>{pauseReplay();seek(Number($('timeline').value));});
$('close-board').addEventListener('click',()=>{$('board-dialog').close();});
$('board-dialog').addEventListener('close',()=>{pauseReplay();dialogMode=null;refresh().catch(e=>notice(e.message));});
$('board').addEventListener('click',event=>{
  if(dialogMode!=='live'||!board)return;
  const rect=$('board').getBoundingClientRect(),x=Math.floor((event.clientX-rect.left)/rect.width*board.width),y=Math.floor((event.clientY-rect.top)/rect.height*board.height);
  if(x<0||y<0||x>=board.width||y>=board.height)return;
  const target=y*board.width+x,sy=Math.floor(selected/board.width),sx=selected%board.width;
  if(selected>=0&&Math.abs(x-sx)+Math.abs(y-sy)===1){const dir=y<sy?0:x>sx?1:y>sy?2:3;move(dir,event.shiftKey);}
  else if(board.owner[y][x]===board.player_id){selected=target;draw();}
  $('board').focus();
});
document.addEventListener('keydown',event=>{
  if(!$('board-dialog').open||dialogMode!=='live'||['INPUT','SELECT','TEXTAREA'].includes(event.target.tagName))return;
  const dirs={ArrowUp:0,w:0,W:0,ArrowRight:1,d:1,D:1,ArrowDown:2,s:2,S:2,ArrowLeft:3,a:3,A:3};
  if(event.key in dirs){event.preventDefault();move(dirs[event.key],event.shiftKey);}
  else if(event.code==='Space'){event.preventDefault();move(0,false,true);}
  else if(event.key.toLowerCase()==='h'){selected=board.general_pos;draw();}
});
for(const button of document.querySelectorAll('[data-dir]'))button.addEventListener('click',()=>move(Number(button.dataset.dir)));
for(const button of document.querySelectorAll('.tab'))button.addEventListener('click',()=>showTab(button.dataset.tab));
$('hero-play').addEventListener('click',()=>{showTab('play');$('play-panel').scrollIntoView({behavior:'smooth',block:'center'});});
$('pool').addEventListener('change',()=>{pool=Number($('pool').value);page=0;renderMeta(meta);loadRank().catch(e=>notice(e.message));if(tab==='history')loadHistory().catch(e=>notice(e.message));});
$('refresh-history').addEventListener('click',()=>loadHistory().catch(e=>notice(e.message)));
$('prev-page').addEventListener('click',()=>{page=Math.max(0,page-1);loadHistory().catch(e=>notice(e.message));});
$('next-page').addEventListener('click',()=>{page++;loadHistory().catch(e=>notice(e.message));});
$('signup-form').addEventListener('submit',async event=>{event.preventDefault();try{const data=await api('/api/players',{name:$('player-name').value},false);token=data.token;player=data.entrant;localStorage.setItem('generals-arena-token',token);notice();await refresh();renderPlayer();}catch(e){notice(e.message);}});
$('recover-form').addEventListener('submit',async event=>{event.preventDefault();token=$('recover-token').value.trim();try{await restorePlayer();localStorage.setItem('generals-arena-token',token);$('recover-token').value='';notice('玩家身份已恢复');await refresh();}catch(e){notice(e.message);}});
$('copy-token').addEventListener('click',async()=>{try{await navigator.clipboard.writeText(token);notice('身份凭证已复制。请保存在你自己的密码管理器中。');}catch(e){notice('浏览器未允许复制凭证。请允许剪贴板权限后重试。');}});
$('start-game').addEventListener('click',()=>openLive().catch(e=>notice(e.message)));
async function refresh(){renderMeta(await api('/api/meta'));if(player&&!$('board-dialog').open){live=await api('/api/play/active');renderPlayer();}await loadRank();if(tab==='history')await loadHistory();$('connection').textContent='联赛在线';}
async function poll(){if(polling)return;polling=true;try{if(player&&dialogMode==='live'&&live&&!live.done&&$('board-dialog').open)updateLive(await api(`/api/play/${live.game_id}`));}catch(e){$('board-error').textContent=e.message;}finally{polling=false;}}
(async()=>{try{await refresh();if(token){await restorePlayer();await loadRank();}}catch(e){notice(e.message);$('connection').textContent='连接异常';}})();
setInterval(poll,150);
setInterval(()=>{if(!$('board-dialog').open)refresh().catch(e=>{$('connection').textContent='连接中断';notice(e.message);});},5000);
