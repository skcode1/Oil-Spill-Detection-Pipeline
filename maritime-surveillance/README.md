# maritime-surveillance — Next.js Dashboard

This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://create-next-app.dev/). It is the frontend that consumes the SIH26143 oil-spill detection & vessel-attribution API (`app/main.py` in the repo root).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the page.

## What it consumes

The dashboard calls the FastAPI backend under `/api/v1`:

- `POST /pipeline` / `POST /pipeline/upload` — detection + drift + attribution (+ optional briefing)
- `POST /drift/hindcast` / `POST /drift/forecast` — forward / backward drift
- `POST /briefing` — NTRO intelligence narrative from a verified dossier JSON
- `GET /health` — model / device / checkpoint readiness

It renders the slick polygons (`spillGeoJson`), suspect vessel points and trajectories (`culpritGeoJson`), and stage status on `SurveillanceMap`.

## Map configuration

The tactical map (`src/SurveillanceMap.tsx`) is intentionally region-agnostic:

- The HUD region label (`regionLabel`) is driven by incident metadata (scene filename / slick location name) and falls back to `"Open Ocean"`.
- The default map center is `[20.0, 0.0]` (neutral open ocean) and recentres automatically to the slick centroid when georeferenced detections are present.
- Basemaps are free and keyless: OpenStreetMap Standard, ESRI Satellite Imagery, and a Tactical Night Mode variant.

This makes the dashboard usable for arbitrary maritime incident locations worldwide rather than a single hardcoded region.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
