# TypeSafe SDK → local Abel

Tested with the official `@typesafe-ai/sdk@0.6.0`. SDK source was reviewed at commit `66880ccded6cb642dc1809620c2b108c33730214`: [typesafe-ai/typesafe-sdk-js](https://github.com/typesafe-ai/typesafe-sdk-js/tree/66880ccded6cb642dc1809620c2b108c33730214).

```bash
npm ci --prefix integrations/typesafe-js --ignore-scripts --no-audit --no-fund
npm run example --prefix integrations/typesafe-js
npm test --prefix integrations/typesafe-js
```

Start the local server first, following the [serving guide](../../docs/SERVING.md). Main client settings:

```js
const client = new TypeSafeClient({
  baseURL: 'http://127.0.0.1:8091',
  apiKey: process.env.JAX_SERVICE_TOKEN || 'local-development',
  defaultModel: 'Jax-1-Abel',
  timeout: 120_000,
  retry: { maxRetries: 0 },
});
```

The `local-development` placeholder satisfies the SDK's API-key requirement. If the server has `JAX_SERVICE_TOKEN` configured, use that exact token. The base URL contains only the root address: the SDK appends `/v1/systemone` or `/v1/models`. Set `defaultModel` explicitly: the SDK otherwise selects Jev, which this server does not provide.

Supported: `systemOne`, `models.list`, `withResponse`, request IDs, HTTP errors and Choice/Score/Noul. See the [serving guide](../../docs/SERVING.md) for limits and the [architecture](../../docs/ARCHITECTURE.md) for numerical semantics. Weights, quality, confidence and performance are not equivalent to Jev. The example calls localhost only. Set `JAX_TEST_URL` to test a different local server address.
