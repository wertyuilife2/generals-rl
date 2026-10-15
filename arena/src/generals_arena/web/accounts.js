// Credentials stay in this browser and are scoped to one league.
export class Accounts {
  constructor(storage, league) {
    this.storage = storage;
    this.key = `generals-arena-accounts:${league}`;
    this.legacyKey = `generals-arena-token:${league}`;
    this.entries = [];
    this.active = null;
    try {
      const saved = JSON.parse(storage.getItem(this.key) || '{}');
      this.entries = (Array.isArray(saved.accounts) ? saved.accounts : []).filter(entry =>
        entry && ['entrant_id', 'name', 'token'].every(key => typeof entry[key] === 'string' && entry[key]));
      this.active = saved.active;
    } catch { /* A malformed list must not prevent restoring a credential. */ }
  }
  get token() {
    return this.entries.find(entry => entry.entrant_id === this.active)?.token
      || this.storage.getItem(this.legacyKey) || '';
  }
  remember(player, token) {
    const account = {entrant_id:player.entrant_id, name:player.name, token};
    const entries = this.entries.filter(entry => entry.entrant_id !== account.entrant_id && entry.token !== token);
    entries.push(account);
    entries.sort((a,b) => a.name.localeCompare(b.name, 'zh-CN'));
    this.storage.setItem(this.key, JSON.stringify({active:account.entrant_id, accounts:entries}));
    this.storage.setItem(this.legacyKey, token);
    this.entries = entries;
    this.active = account.entrant_id;
  }
}
