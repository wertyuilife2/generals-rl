import { test } from 'node:test';
import assert from 'node:assert/strict';
import { historyQuery, placements, nextCondition, entrantColor, compareEntrants } from '../../src/generals_arena/web/history.js';

test('presence events retain multiple included/excluded players and pagination', () => {
  const query = historyQuery('multiplayer', new Map([
    ['a & b', 'include'], ['c', 'include'], ['d', 'exclude'], ['e', 'exclude'], ['f', 'any'],
  ]), 2);
  const decoded = new URLSearchParams(query.toString());
  assert.deepEqual(decoded.getAll('include'), ['a & b', 'c']);
  assert.deepEqual(decoded.getAll('exclude'), ['d', 'e']);
  assert.equal(decoded.get('mode'), 'multiplayer');
  assert.equal(decoded.get('offset'), '40');
  assert.equal(decoded.get('limit'), '20');
  assert.equal(historyQuery('all', new Map(), 0).has('mode'), false);
  assert.equal(historyQuery('1v1', new Map(), 0).get('mode'), '1v1');
});

test('placements use final ranks, shared tie places and policy-level Elo changes', () => {
  const rows = placements({
    entrant_ids: ['a','b','a','c'], ranks: [4,1,2.5,2.5],
    rating_changes: {a:{delta:-4.23}, b:{delta:8.76}, c:{delta:0}},
  });
  assert.deepEqual(rows.map(r => r.seat), [1,2,3,0]);
  assert.deepEqual(rows.map(r => r.place), [1,2,2,4]);
  assert.deepEqual(rows.map(r => r.delta), [8.8,-4.2,0,-4.2]);
  assert.deepEqual(rows.map(r => r.copies), [1,2,1,2]);
});

test('unrated results keep seat order and show no invented ranks or deltas', () => {
  const rows = placements({entrant_ids:['a','b']});
  assert.deepEqual(rows.map(r => r.seat), [0,1]);
  assert.ok(rows.every(r => r.rank === null && r.place === null && r.delta === null));
});

test('text toggles cycle inclusion, exclusion and unrestricted matching', () => {
  let state;
  const conditions = new Map();
  for (const expected of ['include','exclude','any','include']) {
    state = nextCondition(state);
    assert.equal(state, expected);
    conditions.set('a', state);
    const query = historyQuery('all', conditions, 0);
    assert.equal(query.has('include'), state === 'include');
    assert.equal(query.has('exclude'), state === 'exclude');
  }
});

test('versions keep their family hue and become darker without wrapping', () => {
  const versions = [1,2,3,4,5,10,20].map(v => entrantColor({name:`gpt-6-kestrel-v${v}`, kind:'cpp'}));
  assert.equal(new Set(versions.map(c => c.hue)).size, 1);
  assert.equal(new Set(versions.map(c => c.color)).size, versions.length);
  const lightness = versions.map(c=>Number(c.color.match(/ ([\d.]+)%\)$/)[1]));
  assert.ok(lightness.every((v,i)=>i === 0 || v < lightness[i-1]));
  assert.ok(versions.every(c => c.series === 'gpt-6-kestrel'));
  assert.notEqual(versions[0].hue, entrantColor({name:'gpt-6-razor-v1', kind:'cpp'}).hue);
  assert.deepEqual(entrantColor({name:'gpt-6-kestrel-v1', kind:'cpp'}), versions[0]);
  assert.equal(entrantColor({name:'player-v1', kind:'human'}).series, 'player-v1');
});

test('baselines are white, WAIT is gray and sorting uses numeric versions',()=>{
  for (const name of ['random','aggressive-v1','defensive-v1','expansion-v1']) {
    assert.equal(entrantColor({name,kind:'baseline'}).color,'#ffffff');
  }
  assert.equal(entrantColor({name:'WAIT',kind:'cpp'}).color,'#999999');
  const names = ['razor-v10','kestrel-v2','razor-v1','kestrel-v1','razor-v2'];
  assert.deepEqual(names.map(name=>({name,kind:'cpp'})).sort(compareEntrants).map(e=>e.name),
    ['kestrel-v1','kestrel-v2','razor-v1','razor-v2','razor-v10']);
});
