// Empirical win rates; the two rating pools never share observations.
import { compareEntrants, entrantColor } from './history.js';

export function winColor(rate) {
  if (!Number.isFinite(rate)) return 'transparent';
  const low = [126,52,62], middle = [51,58,70], high = [32,126,105];
  const value = Math.max(0,Math.min(1,rate));
  const [from,to,t] = value < .5 ? [low,middle,value*2] : [middle,high,(value-.5)*2];
  return `rgb(${from.map((v,i)=>Math.round(v+(to[i]-v)*t)).join(', ')})`;
}

export function renderMatrix(host, matrix, rows, playerName, openMatchup) {
  const make = (tag,cls,text) => {
    const node = document.createElement(tag); node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  host.replaceChildren();
  if (!matrix?.entrant_ids.length) {
    host.append(make('p','empty','暂无交手数据。'));
    return;
  }
  const names = new Map(rows.map(row=>[row.entrant_id,row.name]));
  const entries = new Map(rows.map(row=>[row.entrant_id,row]));
  const seriesOrder = new Map();
  for (const row of rows) {
    if (Number.isFinite(row.registration_order)) {
      const {series} = entrantColor(row);
      seriesOrder.set(series,Math.min(seriesOrder.get(series) ?? Infinity,row.registration_order));
    }
  }
  // Newer series/versions first, followed by baselines, with WAIT last.
  // Keep source indices so reordering labels never changes a cell's matchup.
  const order = matrix.entrant_ids.map((id,index)=>{
    const entry = entries.get(id) || {name:id};
    const {series,version} = entrantColor(entry);
    const group = series === 'WAIT' ? 2 : entry.kind === 'baseline' ? 1 : 0;
    return {id,index,entry,series,version,group};
  }).sort((a,b)=>a.group-b.group
    || (a.series !== b.series && ((seriesOrder.get(b.series) ?? 0)-(seriesOrder.get(a.series) ?? 0)
      || b.series.localeCompare(a.series,'en',{numeric:true,sensitivity:'base'})))
    || b.version-a.version || compareEntrants(a.entry,b.entry));
  const table = make('table','win-matrix'), head = make('thead',''), header = make('tr','');
  table.setAttribute('aria-label','行选手对列选手的胜率矩阵');
  const corner = make('th','matrix-corner'); corner.scope = 'col'; header.append(corner);
  const columnNodes = order.map(()=>[]), rowNodes = [], positions = new WeakMap();
  let highlighted = new Set();
  const highlight = position => {
    const next = new Set(position ? [...rowNodes[position[0]],...columnNodes[position[1]]] : []);
    for (const node of highlighted) if (!next.has(node)) node.classList.remove('matrix-highlight');
    for (const node of next) if (!highlighted.has(node)) node.classList.add('matrix-highlight');
    highlighted = next;
  };
  table.addEventListener('pointerover',event=>highlight(positions.get(event.target)));
  table.addEventListener('pointerleave',()=>highlight(null));
  for (const {id} of order) {
    const cell = make('th','matrix-column'); cell.scope = 'col';
    columnNodes[header.children.length-1].push(cell);
    cell.append(playerName(id,names.get(id),'matrix-name')); header.append(cell);
  }
  head.append(header); table.append(head);
  const body = make('tbody','');
  order.forEach(({id:left,index:i},r)=>{
    const row = make('tr',''), label = make('th','matrix-row'); label.scope = 'row';
    label.append(playerName(left,names.get(left),'matrix-name')); row.append(label);
    rowNodes.push([label]);
    order.forEach(({id:right,index:j},c)=>{
      const sample = matrix.cells[i][j], cell = make('td','matrix-cell');
      rowNodes[r].push(cell); columnNodes[c].push(cell); positions.set(cell,[r,c]);
      if (sample) {
        cell.textContent = `${Math.round(sample.win_rate*100)}%`;
        cell.style.backgroundColor = winColor(sample.win_rate);
        cell.title = `${names.get(left)} 对 ${names.get(right)}：胜率 ${(sample.win_rate*100).toFixed(1)}% · ${sample.games} 局 · 平局 ${(sample.draw_rate*100).toFixed(1)}%`;
      } else {
        cell.textContent = '—';
        cell.title = left === right ? '同策略不作比较' : `${names.get(left)} 与 ${names.get(right)} 暂无交手`;
      }
      cell.setAttribute('aria-label',cell.title);
      if (openMatchup && left !== right) {
        const button = make('button','matrix-match',cell.textContent);
        button.type = 'button';
        button.setAttribute('aria-label',`${cell.title}；查看对局记录`);
        button.addEventListener('click',()=>openMatchup(left,right));
        positions.set(button,[r,c]);
        cell.textContent = ''; cell.append(button);
      }
      row.append(cell);
    });
    body.append(row);
  });
  const layout = make('div','matrix-layout');
  table.append(body); layout.append(table); host.append(layout);
}
