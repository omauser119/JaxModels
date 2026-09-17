import { TypeSafeClient, choice, noul, score } from '@typesafe-ai/sdk';

const client = new TypeSafeClient({
  baseURL: 'http://127.0.0.1:8091',
  apiKey: process.env.JAX_SERVICE_TOKEN || 'local-development',
  defaultModel: 'Jax-1-Abel',
  timeout: 120_000,
  retry: { maxRetries: 0 },
});
console.log(await client.models.list());
const response = await client.systemOne({
  state: { message: 'I was charged twice. Please refund the duplicate payment.' },
  questions: {
    team: choice('Which team should handle this message?', {
      billing: 'Payments and refunds', technical: 'Software defects', other: 'Everything else',
    }),
    urgent: score('How urgent is the message?', [
      'Routine request without an explicit deadline',
      'Time-sensitive request with an explicit deadline',
      'Immediate emergency blocking a business operation',
    ]),
    refund: noul('Does the message explicitly request a refund?'),
  },
}).withResponse();
console.log(JSON.stringify({ requestId: response.requestId, ...response.data }, null, 2));
