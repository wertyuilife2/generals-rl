import assert from 'node:assert/strict';
import { test } from 'node:test';
import { Accounts } from '../../src/generals_arena/web/accounts.js';

test('multiple accounts survive switching, reload and credential recovery without crossing leagues',()=>{
  const values=new Map(), storage={getItem:key=>values.get(key),setItem:(key,value)=>values.set(key,value)};
  values.set('generals-arena-token:one','old-token');
  const accounts=new Accounts(storage,'one');
  assert.equal(accounts.token,'old-token');
  accounts.remember({entrant_id:'a',name:'Alice'},accounts.token);
  accounts.remember({entrant_id:'b',name:'Bob'},'b-token');
  let restored=new Accounts(storage,'one');
  assert.equal(restored.token,'b-token');
  restored.remember(restored.entries.find(e=>e.entrant_id==='a'),'old-token');
  restored=new Accounts(storage,'one');
  assert.equal(restored.token,'old-token');
  assert.equal(restored.entries.length,2);
  assert.equal(new Accounts(storage,'two').token,'');
  assert.equal(new Accounts(storage,'two').entries.length,0);
  values.set('generals-arena-accounts:one','broken');
  assert.equal(new Accounts(storage,'one').token,'old-token');
});
