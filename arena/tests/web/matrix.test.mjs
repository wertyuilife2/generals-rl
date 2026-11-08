import assert from 'node:assert/strict';
import { test } from 'node:test';
import { winColor } from '../../src/generals_arena/web/matrix.js';

test('matrix colors distinguish no evidence, zero wins and fifty percent',()=>{
  assert.equal(winColor(null),'transparent');
  assert.equal(winColor(NaN),'transparent');
  assert.equal(winColor(0),'rgb(126, 52, 62)');
  assert.equal(winColor(.5),'rgb(51, 58, 70)');
  assert.equal(winColor(1),'rgb(32, 126, 105)');
  assert.equal(winColor(-1),winColor(0));
  assert.equal(winColor(2),winColor(1));
});
