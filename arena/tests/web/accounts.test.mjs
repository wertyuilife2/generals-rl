import assert from 'node:assert/strict';
import { test } from 'node:test';
import { Accounts } from '../../src/generals_arena/web/accounts.js';

const roster = [{entrant_id:'b',name:'Bob',kind:'human'}, {entrant_id:'a',name:'Alice',kind:'human'},
  {entrant_id:'wait',name:'WAIT',kind:'cpp'}];

test('all league accounts are selectable in a fresh browser; only the selection is saved',()=>{
  const values=new Map(), storage={getItem:key=>values.get(key),setItem:(key,value)=>values.set(key,value),removeItem:key=>values.delete(key)};
  const accounts=new Accounts(storage,'one');
  accounts.update(roster);
  assert.deepEqual(accounts.entries.map(entry=>entry.entrant_id),['a','b']);
  assert.equal(accounts.active,'a');
  accounts.remember(roster[0]);
  assert.deepEqual(JSON.parse(values.get('generals-arena-accounts:one')),{active:'b'});
  const restored=new Accounts(storage,'one');
  restored.update(roster);
  assert.equal(restored.active,'b');
  assert.equal(new Accounts(storage,'two').active,'');
  restored.update(roster.slice(1));
  assert.equal(restored.active,'a');
  restored.update([]);
  assert.equal(restored.active,'');
});

test('legacy credentials are discarded while the selected account survives',()=>{
  const values=new Map(), storage={getItem:key=>values.get(key),setItem:(key,value)=>values.set(key,value),removeItem:key=>values.delete(key)};
  values.set('generals-arena-token:one','old-token');
  values.set('generals-arena-accounts:one',JSON.stringify({active:'b',accounts:[{...roster[0],token:'old-token'}]}));
  const accounts=new Accounts(storage,'one');
  accounts.update(roster);
  assert.equal(accounts.active,'b');
  assert.deepEqual(JSON.parse(values.get('generals-arena-accounts:one')),{active:'b'});
  assert.equal(values.has('generals-arena-token:one'),false);
  values.set('generals-arena-accounts:one','broken');
  const malformed=new Accounts(storage,'one');
  malformed.update(roster);
  assert.equal(malformed.active,'a');
});

test('account selection works when browser storage is unavailable',()=>{
  const unavailable=()=>{throw Error('storage unavailable');};
  const accounts=new Accounts({getItem:unavailable,setItem:unavailable,removeItem:unavailable},'one');
  accounts.update(roster);
  accounts.remember(roster[0]);
  assert.equal(accounts.active,'b');
});
