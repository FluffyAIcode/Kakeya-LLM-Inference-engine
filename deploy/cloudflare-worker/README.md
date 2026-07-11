# kakeya.ai Cloudflare Worker

`kakeya-inference-network` owns `kakeya.ai/*` and forwards the public product
surface to the direct origin at `agent.kakeya.ai`.

Routing:

- `/` and `/network` → `/network`
- `/v1/network/*` → same API path
- `/healthz` → `/v1/network/summary`
- unknown browser paths → dashboard

## Validate

```bash
npm install
npm audit --omit=dev
npm run check
```

## Deploy

```bash
npx wrangler login
npm run deploy
```

Expected route:

```text
kakeya.ai/* (zone kakeya.ai)
```

## Verify

```bash
curl -fsS https://kakeya.ai/ | grep "Kakeya Inference Network"
curl -fsS https://kakeya.ai/healthz
curl -fsS https://kakeya.ai/v1/network/nodes
```

Responses carry:

```text
X-Kakeya-Surface: inference-network
```

## Rollback

List versions/deployments:

```bash
npx wrangler versions list
npx wrangler deployments list
```

Roll back with Wrangler's version rollback/deployment command, or remove the
`kakeya.ai/*` route. The direct origin remains available at:

```text
https://agent.kakeya.ai/network
```
