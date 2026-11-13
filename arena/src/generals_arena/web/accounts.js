// Accounts come from the league; the browser only remembers its selection.
export class Accounts {
  constructor(storage, league) {
    this.storage = storage;
    this.key = `generals-arena-accounts:${league}`;
    this.entries = [];
    this.active = '';
    try {
      const saved = JSON.parse(storage.getItem(this.key) || '{}');
      if (typeof saved.active === 'string') this.active = saved.active;
    } catch { /* A malformed preference must not hide the league's accounts. */ }
    // Discard credentials saved by the previous UI, retaining the selected ID.
    this.save();
    try { storage.removeItem(`generals-arena-token:${league}`); } catch {}
  }
  update(entrants) {
    this.entries = entrants.filter(entry => entry.kind === 'human' && !entry.disqualified)
      .map(({entrant_id,name}) => ({entrant_id,name}))
      .sort((a,b) => a.name.localeCompare(b.name, 'zh-CN'));
    if (!this.entries.some(entry => entry.entrant_id === this.active)) {
      this.active = this.entries[0]?.entrant_id || '';
      this.save();
    }
  }
  remember(player) {
    this.entries = [...this.entries.filter(entry => entry.entrant_id !== player.entrant_id),
      {entrant_id:player.entrant_id,name:player.name}].sort((a,b) => a.name.localeCompare(b.name, 'zh-CN'));
    this.active = player.entrant_id;
    this.save();
  }
  save() {
    try { this.storage.setItem(this.key, JSON.stringify({active:this.active})); }
    catch { /* Account selection also works when browser storage is unavailable. */ }
  }
}
