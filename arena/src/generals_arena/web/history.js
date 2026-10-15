// Match history uses AND between presence/absence events; seats remain distinct.
export function historyQuery(mode, conditions, page, limit=20) {
  const query = new URLSearchParams({offset: page * limit, limit});
  if (mode !== 'all') query.set('mode', mode);
  for (const [id, condition] of conditions) {
    if (condition === 'include' || condition === 'exclude') query.append(condition, id);
  }
  return query;
}

export function nextCondition(condition) {
  return condition === 'include' ? 'exclude' : condition === 'exclude' ? 'any' : 'include';
}

// Keep a series' hue stable when entrants are added, retired or reordered.
export function entrantColor({name, kind}) {
  const match = kind === 'human' ? null : name.match(/^(.+)-v(0|[1-9][0-9]*)$/);
  const series = match ? match[1] : name;
  const version = match ? Number(match[2]) : 0;
  if (kind !== 'human' && name.toUpperCase() === 'WAIT')
    return {series:'WAIT', version, hue:0, color:'#999999'};
  if (kind === 'baseline') return {series:'baseline', version, hue:0, color:'#ffffff'};
  let hash = 2166136261;
  for (const character of series.toLowerCase()) hash = Math.imul(hash ^ character.codePointAt(0), 16777619);
  const hue = (hash >>> 0) % 360;
  // New versions become deeper without wrapping or recoloring older versions.
  // The lower bound keeps text readable against the dark panels.
  const lightness = 58 + 28 * Math.exp(-Math.max(0,version-1)/5);
  return {series, version, hue, color: `hsl(${hue} 76% ${lightness}%)`};
}

export function compareEntrants(a,b) {
  const first = entrantColor(a), second = entrantColor(b);
  return first.series.localeCompare(second.series,'en',{numeric:true,sensitivity:'base'})
    || first.version - second.version
    || a.name.localeCompare(b.name,'en',{numeric:true,sensitivity:'base'})
    || (a.entrant_id || '').localeCompare(b.entrant_id || '');
}

export function placements(match) {
  const copies = new Map(), ties = new Map();
  for (const id of match.entrant_ids) copies.set(id, (copies.get(id) || 0) + 1);
  const seats = match.entrant_ids.map((id, seat) => {
    const rank = Number.isFinite(match.ranks?.[seat]) ? match.ranks[seat] : null;
    if (rank !== null) ties.set(rank, (ties.get(rank) || 0) + 1);
    return {id, seat, rank};
  }).sort((a, b) => (a.rank ?? Infinity) - (b.rank ?? Infinity) || a.seat - b.seat);
  return seats.map(entry => {
    const change = match.rating_changes?.[entry.id]?.delta;
    const delta = Number.isFinite(change) ? Math.round(change * 10) / 10 : null;
    // Ratings store average ranks; display the first shared place for a tie.
    const place = entry.rank === null ? null : entry.rank - (ties.get(entry.rank) - 1) / 2;
    return {...entry, place, delta, copies: copies.get(entry.id)};
  });
}
