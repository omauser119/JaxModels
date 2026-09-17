import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { TypeSafeClient, NotFoundError, BadRequestError } from '@typesafe-ai/sdk';
const baseURL = process.env.JAX_TEST_URL || 'http://127.0.0.1:8091';
const client = new TypeSafeClient({baseURL, apiKey: process.env.JAX_SERVICE_TOKEN || 'local-development',
  defaultModel: 'Jax-1-Abel', timeout: 120_000, retry: {maxRetries: 0}});
const models = await client.models.list().withResponse();
assert.equal(models.data[0].name, 'Jax-1-Abel');
assert.ok(models.requestId);
await assert.rejects(client.systemOne({model:'jev-latest', state:'x',questions:{q:{type:'noul',instructions:'Is x present?'}}}),
  error => error instanceof NotFoundError && !!error.requestId);
await assert.rejects(client.systemOne({state:'x',questions:{q:{type:'noul'}}}),
  error => error instanceof BadRequestError && !!error.requestId);
const request = JSON.parse(readFileSync(new URL('../../examples/ticket.json', import.meta.url)));
const result = await client.systemOne(request).withResponse();
assert.equal(result.response.status, 200);
assert.ok(result.requestId);
assert.equal(result.data.model,'Jax-1-Abel');
assert.ok(Number.isInteger(result.data.usage.input_tokens) && result.data.usage.input_tokens > 0);
assert.equal(result.data.usage.output_tokens, 0);
const baseline = JSON.parse(readFileSync(new URL('../../models/Jax-1-Abel/example-response.json', import.meta.url)));
for (const [qid, q] of Object.entries(request.questions)) {
  const answer = result.data.answers[qid];
  assert.equal(answer.type, q.type);
  if (q.type === 'noul') assert.ok(Math.abs(answer.noul - baseline.answers[qid].noul) < 1e-4);
  else {
    assert.ok(Math.abs(Object.values(answer.probabilities).reduce((a,b)=>a+b,0)-1)<1e-8);
    for (const [label, value] of Object.entries(answer.probabilities))
      assert.ok(Math.abs(value - baseline.answers[qid].probabilities[label]) < 1e-4);
  }
}
const report = {passed:true,sdk:'@typesafe-ai/sdk@0.6.0',baseURL,model:'Jax-1-Abel',
  checks:['models.list','request ID header','unknown model rejection','invalid question error','systemOne Choice/Score/Noul','real input token usage','numerical parity with original checkpoint'],
  response:result.data};
mkdirSync(new URL('../../runtime/', import.meta.url), {recursive:true});
writeFileSync(new URL('../../runtime/sdk-compatibility.json', import.meta.url),JSON.stringify(report,null,2)+'\n');
console.log('Official TypeSafe SDK: models, errors, request IDs, all primitives, token usage and checkpoint parity OK');
