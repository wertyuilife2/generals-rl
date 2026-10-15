import { Battlefield } from './board.js';
import { Playback } from './playback.js';
import { MoveInput } from './input.js';
import { historyQuery, placements, nextCondition, entrantColor, compareEntrants } from './history.js';
import { renderMatrix } from './matrix.js';
import { renderQueue } from './queue.js';
import { Accounts } from './accounts.js';
import { ReplayTimeline } from './replay.js';
const $ = id => document.getElementById(id);
const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text !== undefined) n.textContent = text; return n; };
let meta, names = new Map(), colors = new Map(), page = 0, tab = 'ranking', player = null;
let historyMode = 'all', historyRoster = '';
const historyConditions = new Map();
let historyRequest = 0;
let historySignature = '';
let rankingMode = '1v1', rankingRequest = 0;
let matrixSignature = '';
let rankingSignature = '';
let queueSignature = '';
let metaSignature = '';
let tierBoundaries = [];
const tierLayoutObserver = new ResizeObserver(positionTierLabels);
tierLayoutObserver.observe($('ranking-panel'));
let token = '', accounts = null, accountLeague = null, accountBusy = false, showSignup = false;
let accountSignature = '', opponentSignature = '';
let opponentChoices = [];
let live = null, board = null, sending = false;
const outbox = [];
let replay = null, timeline = null, frame = 0, dialogMode = null, polling = false;
let pollController = null;
let replayController = null;
const formatter = new Intl.NumberFormat('zh-CN');
const number = n => formatter.format(n);
function notice(message = '') { $('notice').hidden = !message; $('notice').textContent = message; }
async function api(path, body, auth = true, credential = token, signal = undefined) {
  const headers = {};
  if (credential && auth) headers.Authorization = `Bearer ${credential}`;
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  const response = await fetch(path, {method: body === undefined ? 'GET' : 'POST', headers, body: body === undefined ? undefined : JSON.stringify(body), signal});
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
function playerName(id, name, className) {
  const button = el('button', `entrant-name ${className}`, name);
  button.type = 'button';
  button.style.setProperty('--entrant-color', colors.get(id) || 'var(--accent)');
  button.title = `筛选 ${name} 出现的对局`;
  button.addEventListener('click', () => {
    historyConditions.set(id, 'include'); filterHistory();
    $('history-filter').scrollIntoView({block:'start', behavior:'smooth'});
  });
  return button;
}
function renderMeta(value) {
  const signature = JSON.stringify(value);
  if (signature === metaSignature) return;
  metaSignature = signature;
  meta = value;
  const league = meta.league_id || meta.rules_id;
  if (accountLeague !== league) {
    accountLeague = league;
    accounts = new Accounts(localStorage, league);
    token = accounts.token;
    player = null; live = null;
    accountSignature = '';
  }
  names = new Map(meta.entrants.map(e => [e.entrant_id, e.name]));
  colors = new Map(meta.entrants.map(e => [e.entrant_id, entrantColor(e).color]));
  const options = ['0', ...meta.config.player_counts.map(String)];
  if ([...$('play-players').options].map(n => n.value).join() !== options.join()) {
    const selected = $('play-players').value;
    $('play-players').replaceChildren(...options.map(p => {
      const option = el('option', '', p==='0' ? '随机人数（2–8）' : `${p} 人`);
      option.value=p;
      return option;
    }));
    $('play-players').value=options.includes(selected) ? selected : options[1];
  }
  for (const id of historyConditions.keys()) {
    if (!names.has(id)) { historyConditions.delete(id); page = 0; }
  }
  const roster = JSON.stringify(meta.entrants.map(e => [e.entrant_id, e.name]));
  if (roster !== historyRoster) {
    historyRoster = roster;
    $('history-conditions').replaceChildren();
    const ordered = [...meta.entrants].sort(compareEntrants);
    for (const entry of ordered) {
      const button = el('button', 'history-condition');
      button.type = 'button'; button.dataset.entrant = entry.entrant_id;
      button.style.setProperty('--entrant-color', colors.get(entry.entrant_id));
      const mark = el('span', 'condition-mark'); mark.setAttribute('aria-hidden', 'true');
      button.append(mark, el('span', 'condition-name', entry.name));
      $('history-conditions').append(button);
    }
  }
  renderHistoryFilters();
  const counts=meta.config.player_counts;
  const first=meta.map_ranges[counts[0]],last=meta.map_ranges[counts[counts.length-1]];
  $('map-description').textContent = `宽高按人数随机：${counts[0]} 人 ${first[0]}–${first[1]} 格，${counts[counts.length-1]} 人 ${last[0]}–${last[1]} 格。同一 AI 可有多个独立实例。`;
  $('turn-description').textContent = `每 ${meta.config.tick_seconds} 秒执行至多一步。关闭页面后对局继续，按最终结果计分。`;
  const nextQueueSignature = JSON.stringify([meta.jobs,meta.worker_error]);
  if (nextQueueSignature !== queueSignature) {
    queueSignature = nextQueueSignature;
    renderQueue($('jobs'),$('job-count'),meta.jobs,meta.worker_error,playerName);
  }
  renderPlayer();
}
async function loadRank() {
  const request = ++rankingRequest, mode = rankingMode;
  const data = await api(`/api/leaderboard?mode=${mode}`);
  if (request !== rankingRequest) return;
  const nextSignature = JSON.stringify([mode, data, player?.entrant_id]);
  if (nextSignature === rankingSignature) return;
  rankingSignature = nextSignature;
  $('rating-reference').textContent = mode === 'ffa' ? '均值 1000' : `${names.get(data.anchor) || 'WAIT'} = 0`;
  $('pool-caption').textContent = `${mode === '1v1' ? '1v1 · 双人对局' : 'FFA · 3–8 人对局'} · ${number(data.completed_games)} 局 · ${mode === 'ffa' ? 'Codeforces · 均值 1000' : 'GoR'}`;
  const metrics = mode === '1v1' ? [
    ['duel_win_rate','胜率','获胜局数 ÷ 与其他策略的 1v1 对局数；平局计入分母，自我对战不计入'],
    ['draw_rate','和棋率','未分胜负局数 ÷ 与其他策略的 1v1 对局数'],
    ['mean_win_ticks','获胜步数','仅对获胜对局的 tick 数取平均，包含对手投降；受地图和对手影响'],
    ['mean_decision_ms','决策耗时','每局先计算单步平均决策耗时，再对有计时数据的对局等权平均；不含 Arena / IPC 开销'],
  ] : [
    ['win_rate','胜率','夺冠局数 ÷ 参赛局数；任一同策略席位夺冠记为该策略获胜'],
    ['survival_index','存活指数','（总席位数 − 存活名次）÷（总席位数 − 1）；重复席位先平均，再对各局平均'],
    ['kill_index','击杀指数','亲自击杀的席位数 ÷ 其余席位数；重复席位先平均，再对各局平均'],
  ];
  $('ranking-head').replaceChildren(...[
    ['排名 / 选手',''],['Elo','Elo ± 95% 估计区间半宽'],['▲','当前 Elo 减去同系列上一版本的当前 Elo'],
    ...metrics.map(([,label,title])=>[label,title]),
  ].map(([label,title])=>{const th=el('th','',label);th.scope='col';th.title=title;return th;}));
  $('ranking').replaceChildren();
  $('rank-tier-labels').replaceChildren();
  tierBoundaries = [];
  let previousTier = null, previousTierRow = null, rankedPosition = 0;
  data.ranking.forEach(row => {
    const tr = el('tr'), name = el('td'), wrap = el('div', 'rank-player'), detail = el('div'), heading = el('div', 'rank-player-heading');
    heading.append(playerName(row.entrant_id, row.name, 'player-name'));
    detail.append(heading);
    const tier = row.rating_title, unrankedHuman = mode === '1v1' && row.kind === 'human';
    if (unrankedHuman) {
      detail.append(el('div','player-sub','人类'));
    } else if (mode === '1v1' && tier) {
      const badge = titleBadge(tier);
      badge.title = tier.key === 'unranked' ? '人类玩家不参与评级分段'
        : tier.key === 'unrated' ? `还需 ${tier.remaining_games} 场有效对局完成定级`
        : tier.maximum === null ? '已达到最高等级'
          : `距离下一等级还差 ${(tier.maximum-row.elo).toFixed(1)} 分`;
      heading.append(badge);
      const context = [row.kind === 'human' ? '人类' : '',row.anchor ? '零点' : '',row.disqualified ? '已停用' : ''].filter(Boolean);
      if (context.length) detail.append(el('div','player-sub',context.join(' · ')));
      if (!['unranked','unrated'].includes(tier.key)) {
        if (previousTier && previousTier.key !== tier.key) {
          previousTierRow.className = 'rank-tier-end';
          previousTierRow.style.setProperty('--title-color',previousTier.color);
          const label = el('div','rank-tier-label');
          label.style.setProperty('--title-color',previousTier.color);
          label.append(el('span','',previousTier.label),el('span','rank-tier-score',
            previousTier.minimum === null ? `< ${previousTier.maximum}` : number(previousTier.minimum)));
          $('rank-tier-labels').append(label);
          tierBoundaries.push({row:previousTierRow,label});
        }
        previousTier = tier;
        previousTierRow = tr;
      }
    } else {
      detail.append(el('div', 'player-sub', [row.kind === 'human' ? '人类玩家' : row.kind === 'baseline' || row.anchor ? 'Baseline' : 'Algorithm', row.disqualified ? '策略违规 · 已停用' : row.anchor ? '零分基准' : row.provisional ? '定级中' : '已定级'].join(' · ')));
    }
    const place = unrankedHuman ? null : ++rankedPosition;
    wrap.append(el('span', `rank-no${place && place <= 3 ? ' top' : ''}`, place === null ? '' : String(place).padStart(2, '0')), detail); name.append(wrap);
    const change = Number.isFinite(row.elo_delta) ? Math.round(row.elo_delta * 10) / 10 : null;
    const delta = el('td', `elo-delta${change > 0 ? ' positive' : change < 0 ? ' negative' : ''}`,
      change === null ? '—' : `${change > 0 ? '+' : ''}${change.toFixed(1)}`);
    delta.title = row.previous_version ? `相对 ${row.previous_version} 的当前 Elo${change === null ? '（双方均完成对局后显示）' : ''}` : '暂无可比较的上一版本';
    const elo = el('td', 'elo'), estimate = el('span', 'elo-value', row.elo.toFixed(1));
    const ci = el('span', 'elo-ci', row.elo_ci ? `±${row.elo_ci.half_width.toFixed(1)}` : '± —');
    ci.title = row.anchor ? '归一化参考：自身与自身的分差恒为 0；WAIT 的原始评分正常结算' : row.elo_ci
      ? `95% 估计区间 [${row.elo_ci.lower.toFixed(1)}, ${row.elo_ci.upper.toFixed(1)}] · ${row.ci_blocks} 组独立地图；基于名次模型的近似估计`
      : '95% 估计区间：独立地图样本不足，或对手关系尚未连通';
    ci.setAttribute('aria-label', ci.title);
    elo.append(estimate, ci);
    tr.append(name, elo, delta, ...metrics.map(([key])=>{
      const value=row[key];
      const text = !Number.isFinite(value) ? '—' : key === 'mean_win_ticks' ? `${number(Math.round(value))} 步`
        : key === 'mean_decision_ms' ? `${value < 1 ? (value*1000).toFixed(1)+' μs' : value.toFixed(2)+' ms'}`
        : `${(100*value).toFixed(1)}%`;
      const cell=el('td','metric',text);
      if (mode === '1v1') cell.title = key === 'mean_win_ticks' ? `${row.duel_wins ?? 0} 场获胜对局`
        : key === 'mean_decision_ms' ? `${row.timed_games ?? 0} 场有计时记录；缺失数据不记作 0`
        : `${row.duel_games ?? 0} 场与其他策略的 1v1 对局`;
      return cell;
    }));
    $('ranking').append(tr);
    if (player && row.entrant_id === player.entrant_id) $('player-rating').textContent = `${mode.toUpperCase()} · ${row.elo.toFixed(1)} Elo · ${row.games} 局`;
  });
  positionTierLabels();
  const signature = JSON.stringify([mode,data.win_matrix,data.ranking.map(row=>[row.entrant_id,row.name,row.kind,row.registration_order])]);
  if (signature !== matrixSignature) {
    matrixSignature = signature;
    $('matrix-caption').textContent = mode === '1v1'
      ? '行选手战胜列选手的比例 · 平局单列，悬停查看样本数'
      : '行选手最终名次领先列选手的比例 · 重复席位先取平均，平局单列';
    renderMatrix($('win-matrix'),data.win_matrix,data.ranking,playerName,(left,right)=>{
      historyConditions.clear();
      historyConditions.set(left,'include'); historyConditions.set(right,'include');
      historyMode = mode;
      filterHistory();
      $('history-filter').scrollIntoView({block:'start',behavior:'smooth'});
    });
  }
}
function positionTierLabels() {
  const panel = $('ranking-panel');
  if (!tierBoundaries.length || panel.hidden) return;
  const top = panel.getBoundingClientRect().top + panel.clientTop;
  const positions = tierBoundaries.map(({row})=>row.getBoundingClientRect().bottom-top);
  tierBoundaries.forEach(({label},i)=>{label.style.top = `${positions[i]}px`;});
}
function titleBadge(tier) {
  const badge=el('span',`rating-title ${tier.key}`,tier.label);
  badge.style.setProperty('--title-color',tier.color);
  return badge;
}
async function loadHistory() {
  const request = ++historyRequest;
  const query = historyQuery(historyMode, historyConditions, page);
  const filtered = historyMode !== 'all' || historyConditions.size > 0;
  const data = await api(`/api/matches?${query}`);
  if (request !== historyRequest) return; // A slower previous filter must not replace the latest results.
  const lastPage = Math.max(0, Math.ceil(data.total / 20) - 1);
  if (page > lastPage) { page = lastPage; return loadHistory(); }
  const signature = JSON.stringify([query.toString(), historyRoster, data]);
  if (signature === historySignature) return;
  historySignature = signature;
  $('history').replaceChildren();
  for (const match of data.matches) {
    const row = el('article', 'match-row'), summary = el('div', 'match-summary');
    const date = new Date(match.finished * 1000), time = el('time', 'match-time');
    time.dateTime = date.toISOString();
    time.append(el('span', '', date.toLocaleDateString('zh-CN')), el('span', '', date.toLocaleTimeString('zh-CN', {hour12:false})));
    summary.append(time, el('span', 'match-steps', `${match.ticks ?? '—'} 步 · ${match.entrant_ids.length === 2 ? '1v1' : 'FFA'}`));
    if (match.status !== 'complete') {
      const status = el('span', 'match-unrated', '未计分');
      status.title = match.status === 'failed' ? match.policy_error || '执行失败' : '服务中断';
      summary.append(status);
    }
    if (['complete', 'failed'].includes(match.status)) {
      const open = el('button', 'match-open'); open.type = 'button';
      open.setAttribute('aria-label', `查看 ${date.toLocaleString('zh-CN', {hour12:false})} 的对局回放`);
      open.addEventListener('click', () => openReplay(match.game_id).catch(e => notice(e.message)));
      row.append(open);
    }
    const players = el('ol', 'match-placements'); players.setAttribute('aria-label', '本局名次与 Elo 变化');
    for (const entry of placements(match)) {
      const item = el('li', 'match-placement');
      const rank = el('span', 'match-place', entry.place === null ? '—' : `#${entry.place}`);
      const participant = playerName(entry.id, names.get(entry.id) || entry.id.slice(0,8), 'match-player-name');
      const delta = el('span', `match-delta elo-delta${entry.delta > 0 ? ' positive' : entry.delta < 0 ? ' negative' : ''}`,
        entry.delta === null ? '—' : `${entry.delta > 0 ? '+' : ''}${entry.delta.toFixed(1)}`);
      const pool = match.entrant_ids.length === 2 ? '1v1' : 'FFA';
      delta.title = entry.copies > 1 ? `${pool}：该策略本局 ${entry.copies} 个席位共用的 Elo 总变化，不按席位累加` : `本局 ${pool} Elo 变化`;
      if (pool === '1v1' && match.rating_policy === 'gor-relative-codeforces-v9')
        delta.title += '（本局原始评分变化，不含 WAIT 零点的平移）';
      if (entry.delta !== null) delta.append(el('small', '', entry.copies > 1 ? 'Elo 合计' : 'Elo'));
      item.append(rank, participant, delta); players.append(item);
    }
    row.append(summary, players); $('history').append(row);
  }
  if (!data.matches.length) $('history').append(el('p', 'empty', filtered ? '没有符合筛选条件的对局。' : '还没有对局记录。登记算法或开始你的第一局。'));
  $('page-info').textContent = `第 ${page + 1} 页 · 共 ${data.total} 局`;
  $('prev-page').disabled = page === 0; $('next-page').disabled = (page + 1) * 20 >= data.total;
}
function renderPlayer() {
  $('signup').hidden = !!player && !showSignup; $('player-lobby').hidden = !player;
  $('cancel-signup').hidden = !player;
  $('account-controls').hidden = !accounts?.entries.length;
  const signature = JSON.stringify(accounts?.entries.map(({entrant_id,name})=>[entrant_id,name]));
  if (signature !== accountSignature) {
    accountSignature = signature;
    $('play-account').replaceChildren(...(accounts?.entries || []).map(entry => {
      const option=el('option','',entry.name);option.value=entry.entrant_id;return option;
    }));
  }
  const selectedAccount = player?.entrant_id || accounts?.active || '';
  if ($('play-account').value !== selectedAccount) $('play-account').value = selectedAccount;
  for (const id of ['play-account','new-account','create-account','cancel-signup']) $(id).disabled=accountBusy;
  if (player && $('player-label').textContent !== player.name) {
    $('player-label').textContent = player.name; $('play-title').textContent = '准备好下一场对局';
  }
  if (meta) {
    const bots = meta.entrants.filter(e => e.kind !== 'human' && !e.disqualified).length;
    $('start-game').disabled = accountBusy || (!live || live.done) && bots === 0;
    const players = Number($('play-players').value);
    const label = live && !live.done ? '返回当前对局' : bots === 0 ? '需要一个算法对手' : players ? `开始 ${players} 人对局` : '随机匹配并开始';
    if ($('start-game').textContent !== label) $('start-game').textContent = label;
    renderOpponents();
  }
}
function renderOpponents() {
  const bots = meta.entrants.filter(e=>e.kind !== 'human' && !e.disqualified).sort(compareEntrants);
  const players = Number($('play-players').value), active = !!live && !live.done;
  const signature = JSON.stringify([players,bots.map(e=>[e.entrant_id,e.name]),active,accountBusy]);
  $('play-players').disabled = active || accountBusy;
  if (signature === opponentSignature) return;
  opponentSignature = signature;
  const ids = new Set(bots.map(e=>e.entrant_id));
  opponentChoices = Array.from({length:Math.max(0,players-1)},(_,i)=>ids.has(opponentChoices[i])?opponentChoices[i]:null);
  $('play-opponents').replaceChildren(...opponentChoices.map((choice,index)=>{
    const label=el('label','opponent-slot'),select=el('select');
    label.append(el('span','',`对手 ${index+1}`),select);
    const random=el('option','','随机匹配');random.value='';select.append(random);
    for (const bot of bots) {const option=el('option','',bot.name);option.value=bot.entrant_id;select.append(option);}
    select.value=choice || '';select.disabled=active || accountBusy;
    select.addEventListener('change',()=>{opponentChoices[index]=select.value || null;});
    return label;
  }));
  $('opponents-hint').textContent = active ? '当前账号有进行中的对局，结束后可重新选择对手。'
    : players ? '逐个选择算法，也可保留随机匹配。同一算法可重复出场。' : '人数与对手随机；选择固定人数后，可以指定每个对手。';
}
async function restorePlayer(credential = token) {
  if (!credential || accountBusy) return;
  accountBusy=true;renderPlayer();
  try {
    const [nextPlayer,nextLive] = await Promise.all([
      api('/api/me',undefined,true,credential), api('/api/play/active',undefined,true,credential),
    ]);
    activatePlayer(nextPlayer,nextLive,credential);
  } finally {accountBusy=false;renderPlayer();}
}
function activatePlayer(nextPlayer,nextLive,credential) {
  accounts.remember(nextPlayer,credential);
  pollController?.abort();
  token=credential;player=nextPlayer;live=nextLive;
  input.reset();outbox.length=0;showSignup=false;
  $('player-rating').textContent='';
}
function showDialog(mode) {
  if (mode === 'live') { replayController?.abort(); replayController = null; }
  setReplayLoading(false);
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
  $('board-status').textContent=live.error?`本局未计分：${live.error}`:live.done?`对局结束 · ${live.result.winner_seat==null?'达到时限，存活者并列':`${live.entrants[live.result.winner_seat].name} 获胜`} · 已计入 ${live.entrants.length === 2 ? '1v1' : 'FFA'} 排行榜`:`Tick ${board.tick} / ${live.max_ticks} · ${board.alive[board.player_id]?'局部视野 · 输入立即排队，每 tick 执行一步':'你已被淘汰，等待对局结束后计分'}`;
  draw();renderGamePlayers(live.entrants);
  if(live.done)renderPlayer();
}
async function openLive() {
  notice();
  if (!player) { showTab('play'); return; }
  $('start-game').disabled = true;
  try {
    const players = Number($('play-players').value)||null;
    const value = await api('/api/play', {players, opponents:players ? opponentChoices : null});
    const fresh=live?.game_id!==value.game_id || !input.board;
    if(fresh)input.reset();
    live=value;showDialog('live');updateLive(value,true);
    if(fresh)input.home();
    poll();
  } finally { renderPlayer(); }
}
const input = new MoveInput(()=>draw(),command=>{
  outbox.push({gameId:live.game_id,credential:token,command});flushCommands();
});
async function flushCommands() {
  if(sending)return;
  sending=true;
  try {
    while(outbox.length) {
      const pending=outbox[0], {gameId,credential,command}=pending;
      try {
        const value=await api(`/api/play/${gameId}/commands`,command,true,credential);
        if(outbox[0]===pending)outbox.shift();
        if(token!==credential || live?.game_id!==gameId)continue;
        updateLive(value);
        $('board-error').textContent='';
      } catch(error) {
        if(token!==credential || live?.game_id!==gameId){if(outbox[0]===pending)outbox.shift();continue;}
        $('board-error').textContent=error.status?error.message:'连接中断，正在重试操作…';
        if(!error.status){setTimeout(flushCommands,300);break;}
        outbox.length=0;input.pending=[];
        const value=await api(`/api/play/${gameId}`,undefined,true,credential);
        if(token!==credential || live?.game_id!==gameId)continue;
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
  replayController?.abort();
  const controller = replayController = new AbortController();
  replay = timeline = board = null; battlefield.board = null;
  playerNames = ''; $('game-players').replaceChildren();
  showDialog('replay'); setReplayLoading(true); notice();
  $('board-title').textContent = `对局回放 · ${id.slice(0,8)}`;
  $('board-status').textContent = '';
  $('turn-counter').textContent = '—'; $('replay-tick').textContent = '';
  try {
    const data = await api(`/api/replay/${id}`, undefined, false, '', controller.signal);
    if (controller !== replayController) return;
    replay = data; timeline = new ReplayTimeline(data.frames); frame = 0;
    $('timeline').max = data.frames.length - 1;
    $('board-title').textContent = `${data.result.map.players} 人对局回放 · ${id.slice(0,8)}`;
    $('board-status').textContent = `${data.result.map.width} × ${data.result.map.height} · ${data.result.policy_error ? '策略违规中断 · 未计分' : data.result.reason === 'surrender' ? '投降结算' : data.result.reason === 'capture' ? '主城占领' : '达到时限'} · 已校验全部已执行动作与状态`;
    setReplayLoading(false); seek(0); $('board').focus();
  } catch (error) {
    if (controller !== replayController || error.name === 'AbortError') return;
    setReplayLoading(true);
    $('board-dialog').setAttribute('aria-busy', 'false');
    $('replay-loading').textContent = `回放加载失败：${error.message}`;
  } finally {
    if (controller === replayController) replayController = null;
  }
}
function setReplayLoading(loading) {
  $('board-dialog').setAttribute('aria-busy', String(loading));
  $('board').hidden = loading;
  $('replay-loading').hidden = !loading;
  $('replay-loading').textContent = '正在读取并校验回放…';
  for (const id of ['toggle-replay', 'timeline', 'replay-speed']) $(id).disabled = loading;
}
function seek(next) {
  board=timeline.seek(next);frame=timeline.index;
  $('timeline').value=frame;$('replay-tick').textContent=`Tick ${board.tick} / ${replay.result.ticks}`;
  draw();renderGamePlayers(replay.result.entrant_ids.map(id=>({name:names.get(id)||id.slice(0,8)})));
}
const playback = new Playback({
  interval: () => (meta?.config.tick_seconds || .5) * 1000 / Number($('replay-speed').value),
  step: () => { seek(frame + 1); return frame < replay.frames.length - 1; },
  stopped: () => { $('toggle-replay').textContent = '播放'; },
});
function pauseReplay(){playback.stop();}
$('toggle-replay').addEventListener('click',()=>{
  if (!replay || !timeline) return;
  if(playback.running){pauseReplay();return;}if(frame===replay.frames.length-1)seek(0);
  $('toggle-replay').textContent='暂停';playback.start();
});
$('timeline').addEventListener('input',()=>{if(timeline){pauseReplay();seek(Number($('timeline').value));}});
$('close-board').addEventListener('click',()=>{$('board-dialog').close();});
$('board-dialog').addEventListener('close',()=>{pauseReplay();pollController?.abort();replayController?.abort();replayController=null;dialogMode=null;replay=null;timeline=null;refresh().catch(e=>notice(e.message));});
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
for (const button of document.querySelectorAll('[data-ranking-mode]')) button.addEventListener('click', () => {
  rankingMode = button.dataset.rankingMode;
  for (const item of document.querySelectorAll('[data-ranking-mode]')) item.setAttribute('aria-pressed', String(item === button));
  loadRank().catch(e => notice(e.message));
});
function filterHistory() {
  page = 0;
  renderHistoryFilters();
  if (tab !== 'history') showTab('history');
  else loadHistory().catch(e => notice(e.message));
}
function renderHistoryFilters() {
  $('clear-history-filters').disabled = historyMode === 'all' && !historyConditions.size;
  for (const button of document.querySelectorAll('[data-history-mode]')) button.setAttribute('aria-pressed', String(button.dataset.historyMode === historyMode));
  for (const button of $('history-conditions').children) {
    const condition = historyConditions.get(button.dataset.entrant) || 'any';
    button.dataset.condition = condition;
    button.setAttribute('aria-pressed', condition === 'include' ? 'true' : condition === 'exclude' ? 'mixed' : 'false');
    const state = {any:'不限', include:'出现', exclude:'不出现'};
    button.setAttribute('aria-label', `${names.get(button.dataset.entrant)}：${state[condition]}，点击改为${state[nextCondition(condition)]}`);
    button.title = button.getAttribute('aria-label');
    button.firstElementChild.textContent = {any:'·', include:'+', exclude:'−'}[condition];
  }
}
for (const button of document.querySelectorAll('[data-history-mode]')) button.addEventListener('click', () => { historyMode = button.dataset.historyMode; filterHistory(); });
$('history-conditions').addEventListener('click', event => {
  const button = event.target.closest('button[data-entrant]');
  if (!button) return;
  const id = button.dataset.entrant, next = nextCondition(historyConditions.get(id));
  if (next === 'any') historyConditions.delete(id);
  else historyConditions.set(id, next);
  filterHistory();
});
$('history-filter').addEventListener('submit', event => { event.preventDefault(); filterHistory(); });
$('clear-history-filters').addEventListener('click', () => {
  historyMode = 'all'; historyConditions.clear();
  filterHistory();
});
$('prev-page').addEventListener('click',()=>{page=Math.max(0,page-1);loadHistory().catch(e=>notice(e.message));});
$('next-page').addEventListener('click',()=>{page++;loadHistory().catch(e=>notice(e.message));});
$('new-account').addEventListener('click',()=>{showSignup=true;renderPlayer();$('player-name').focus();});
$('cancel-signup').addEventListener('click',()=>{showSignup=false;renderPlayer();});
$('play-account').addEventListener('change',async()=>{
  const account=accounts.entries.find(entry=>entry.entrant_id===$('play-account').value);
  if(!account || account.entrant_id===player?.entrant_id)return;
  try{await restorePlayer(account.token);notice();await loadRank();}catch(e){notice(e.message);}
});
$('play-players').addEventListener('change',renderPlayer);
$('signup-form').addEventListener('submit',async event=>{
  event.preventDefault();if(accountBusy)return;accountBusy=true;renderPlayer();
  try{
    const data=await api('/api/players',{name:$('player-name').value},false);
    activatePlayer(data.entrant,null,data.token);$('player-name').value='';notice();await refresh();
  }catch(e){notice(e.message);}finally{accountBusy=false;renderPlayer();}
});
$('recover-form').addEventListener('submit',async event=>{
  event.preventDefault();if(accountBusy)return;
  try{await restorePlayer($('recover-token').value.trim());$('recover-token').value='';notice('账号已恢复并加入列表');await refresh();}
  catch(e){notice(e.message);}
});
$('copy-token').addEventListener('click',async()=>{try{await navigator.clipboard.writeText(token);notice('身份凭证已复制。请保存在你自己的密码管理器中。');}catch(e){notice('浏览器未允许复制凭证。请允许剪贴板权限后重试。');}});
$('start-game').addEventListener('click',()=>openLive().catch(e=>notice(e.message)));
async function refresh(){
  renderMeta(await api('/api/meta'));
  if(player&&!accountBusy&&!$('board-dialog').open){
    const credential=token,value=await api('/api/play/active',undefined,true,credential);
    if(token===credential){live=value;renderPlayer();}
  }
  await loadRank();if(tab==='history')await loadHistory();
  if($('connection').textContent!=='联赛在线')$('connection').textContent='联赛在线';
}
async function poll(){
  if(polling)return;
  if(!(player&&dialogMode==='live'&&live&&!live.done&&$('board-dialog').open))return;
  polling=true;
  const id=live.game_id,credential=token,controller=new AbortController();pollController=controller;
  try{const value=await api(`/api/play/${id}?after=${Math.max(0,input.revision)}`,undefined,true,credential,controller.signal);if(token===credential&&live?.game_id===id)updateLive(value);}
  catch(e){if(e.name!=='AbortError'&&token===credential&&live?.game_id===id)$('board-error').textContent=e.message;}
  finally{if(pollController===controller)pollController=null;polling=false;setTimeout(poll,50);}
}
(async()=>{
  try{await refresh();}catch(e){notice(e.message);$('connection').textContent='连接异常';return;}
  if(token){
    try{await restorePlayer();await loadRank();}
    catch(e){notice(e.status===403?'保存的玩家凭证无法用于此联赛，请恢复身份或创建玩家。':e.message);renderPlayer();}
  }
})();
setInterval(poll,1000);
setInterval(()=>{if(!document.hidden&&!$('board-dialog').open)refresh().catch(e=>{$('connection').textContent='连接中断';notice(e.message);});},5000);
