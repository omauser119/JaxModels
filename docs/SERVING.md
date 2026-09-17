# Local TypeSafe-compatible server

Install the hash-pinned Python dependencies and run `scripts/serve-abel --port 8091` as in the README. The server starts a persistent native worker and loads the bundled checkpoint. It does not call TypeSafe or a teacher API during inference.

```js
import { TypeSafeClient } from '@typesafe-ai/sdk';
const client = new TypeSafeClient({
  baseURL: 'http://127.0.0.1:8091', // no /v1 suffix
  apiKey: process.env.JAX_SERVICE_TOKEN || 'local-development',
  defaultModel: 'Jax-1-Abel',
  timeout: 120000,
  retry: { maxRetries: 0 },
});
const result = await client.systemOne({
  state: { text: 'Please cancel my reservation.' },
  questions: {
    cancel: { type: 'noul', instructions: 'Is the user explicitly requesting cancellation?' }
  }
});
console.log(result.answers.cancel.noul);
```

Pinned JS integration: `integrations/typesafe-js/package-lock.json` (`@typesafe-ai/sdk@0.6.0`). `npm test` verifies model listing, typed errors, request IDs, token accounting, all three primitives and numerical parity with the released head. The compatibility surface is tested, not every SDK feature or future SDK version. Jev model aliases are deliberately rejected; use `Jax-1-Abel`.

| Route | Purpose |
|---|---|
| POST `/v1/systemone` | SDK-compatible decisions |
| GET `/v1/models` | SDK-compatible model listing |
| POST `/v1/decide` | Jax decision interface |
| GET `/v1/model` | Model metadata |
| GET `/healthz`, `/readyz` | Health and readiness |
| GET `/metrics` | Service metrics |

One worker/context, no unbounded inference queue; busy requests receive 429. Limits: 256 KiB request body, 16 questions, 32 judgments, 10-second body deadline and 120-second inference deadline. CPU latency scales with state length and number of criteria. Output tokens are zero because decisions are numeric; input usage counts repeated prefills. Core Choice supports 2–255 alternatives, but the HTTP judgment budget is stricter.

Loopback access is intended for local applications. A non-loopback binding requires `JAX_SERVICE_TOKEN` of at least 32 characters, supplied by clients as a bearer token. Set up TLS and deployment ingress separately. This prototype's bounded serving behavior does not establish production model quality or operational capacity. Do not enable multiple worker processes casually: each needs its own model memory.
