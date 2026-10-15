// Compact current work, with completed jobs available in an expandable list.
export function renderQueue(host,badge,jobs,workerError,playerName) {
  const make = (tag,cls,text) => {
    const node = document.createElement(tag); node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const number = n => new Intl.NumberFormat('zh-CN').format(n);
  const expanded = host.querySelector?.('.queue-completed')?.open || false;
  const complete = jobs.filter(j=>j.status === 'complete');
  const unfinished = jobs.filter(j=>j.status !== 'complete');
  const failed = unfinished.filter(j=>j.status === 'failed').length;
  badge.textContent = failed ? `${failed} 项异常` : unfinished.length ? `${unfinished.length} 待完成` : jobs.length ? '全部完成' : '空闲';
  host.replaceChildren();
  if (!jobs.length) {
    host.append(make('p','empty','暂无评测任务。'));
  } else {
    const overview = make('div','queue-overview');
    for (const [label,value] of [
      ['任务已完成',`${number(complete.length)} / ${number(jobs.length)}`],
      ['对局已完成',number(jobs.reduce((sum,j)=>sum+j.done,0))],
    ]) {
      const item = make('div','queue-total');
      item.append(make('strong','',value),make('span','',label)); overview.append(item);
    }
    host.append(overview);
    if (!unfinished.length) host.append(make('p','queue-idle','✓ 全部评测已完成'));
    const priority = j=>j.status === 'failed' ? 0 : j.done > 0 ? 1 : 2;
    for (const job of [...unfinished].sort((a,b)=>priority(a)-priority(b) || a.created-b.created)) {
      const state = job.status === 'failed' ? 'failed' : job.done > 0 ? 'started' : 'waiting';
      const card = make('article',`queue-job ${state}`), header = make('div','queue-job-header');
      header.append(playerName(job.entrant_id,job.name,'queue-name'),
        make('span','queue-state',{failed:'异常',started:'已开始',waiting:'待评测'}[state]));
      card.append(header);
      const modes = Object.keys(job.modes || {}).length ? job.modes : {'总计':job};
      for (const [mode,value] of Object.entries(modes)) {
        const row = make('div',`queue-mode ${mode}`), label = make('div','queue-mode-label');
        label.append(make('span','',mode.toUpperCase()),make('span','',`${number(value.done)} / ${number(value.budget)}`));
        const bar = make('progress',''); bar.max = value.budget; bar.value = value.done;
        bar.setAttribute('aria-label',`${job.name} ${mode.toUpperCase()} 已完成 ${value.done} / ${value.budget} 局`);
        row.append(label,bar); card.append(row);
      }
      if (job.error) card.append(make('p','queue-error',job.error));
      host.append(card);
    }
    if (complete.length) {
      const archive = make('details','queue-completed'); archive.open = expanded;
      archive.append(make('summary','',`已完成记录 · ${number(complete.length)} 项`));
      const list = make('div','queue-completed-list');
      for (const job of [...complete].sort((a,b)=>b.created-a.created)) {
        const row = make('div','queue-completed-row');
        const modes = Object.entries(job.modes || {});
        row.append(playerName(job.entrant_id,job.name,'queue-name'),make('small','',modes.length
          ? modes.map(([mode,value])=>`${mode.toUpperCase()} ${number(value.done)} 局`).join(' · ')
          : `${number(job.done)} 局`));
        list.append(row);
      }
      archive.append(list); host.append(archive);
    }
  }
  if (workerError) host.append(make('p','queue-error',workerError));
}
