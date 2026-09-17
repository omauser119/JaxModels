# TypeSafe SDK → локальный Abel

Проверено с официальным `@typesafe-ai/sdk@0.6.0`. Исходники исследованы на commit `66880ccded6cb642dc1809620c2b108c33730214` [typesafe-ai/typesafe-sdk-js](https://github.com/typesafe-ai/typesafe-sdk-js/tree/66880ccded6cb642dc1809620c2b108c33730214).

```bash
npm ci --prefix integrations/typesafe-js --ignore-scripts --no-audit --no-fund
npm run example --prefix integrations/typesafe-js
npm test --prefix integrations/typesafe-js
```

Основные параметры:

```js
const client = new TypeSafeClient({
  baseURL: 'http://127.0.0.1:8091',
  apiKey: process.env.JAX_SERVICE_TOKEN || 'local-development',
  defaultModel: 'Jax-1-Abel',
  timeout: 120_000,
  retry: { maxRetries: 0 },
});
```

Локальный ключ `local-development` удовлетворяет требованию SDK. Если сервер запущен с `JAX_SERVICE_TOKEN`, нужен именно этот токен. Адрес включает только корень: SDK сам добавляет `/v1/systemone` или `/v1/models`. Не оставляйте defaultModel по умолчанию: SDK выбирает Jev, которого наш сервер не предоставляет.

Поддержаны `systemOne`, `models.list`, `withResponse`, request ID, ошибки HTTP, Choice/Score/Noul. Лимиты и численные semantics — [профиль Abel](../../docs/PRODUCTION.md). Веса, качество, confidence и производительность не тождественны Jev. Пример вызывает только localhost.
