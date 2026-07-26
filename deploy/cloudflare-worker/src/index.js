import { PAGE } from "./page.js";

const ORIGIN = "https://agent.kakeya.ai";
const BUILD_ID = "2026.07.25-frontend-runtime-r2";

function pageResponse(request) {
  return new Response(request.method === "HEAD" ? null : PAGE, {
    headers: {
      "Content-Type": "text/html; charset=utf-8",
      "Cache-Control": "private, no-store, no-cache, must-revalidate, max-age=0",
      "CDN-Cache-Control": "no-store",
      "Cloudflare-CDN-Cache-Control": "no-store",
      "Surrogate-Control": "no-store",
      "Pragma": "no-cache",
      "Expires": "0",
      "Content-Security-Policy": "default-src 'self'; script-src 'unsafe-inline'; style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; connect-src 'self'; img-src 'self' data:; font-src 'self' data:; object-src 'none'; base-uri 'self'; frame-ancestors 'none'",
      "X-Kakeya-Surface": "proof-observatory",
      "X-Kakeya-Build": BUILD_ID,
      "X-Content-Type-Options": "nosniff",
      "Referrer-Policy": "same-origin",
    },
  });
}

export default {
  async fetch(request) {
    const incoming = new URL(request.url);
    let path = incoming.pathname;

    if (path === "/" || path === "/index.html" || path === "/network") {
      return pageResponse(request);
    } else if (path === "/healthz") {
      path = "/v1/network/summary";
    } else if (
      !path.startsWith("/v1/network/") &&
      !path.startsWith("/v1/proof/")
    ) {
      return pageResponse(request);
    }

    const target = new URL(path + incoming.search, ORIGIN);
    const requestHeaders = new Headers(request.headers);
    requestHeaders.delete("host");
    if (path.startsWith("/v1/proof/")) {
      requestHeaders.set("Cache-Control", "no-cache");
    }
    const upstreamRequest = new Request(target.toString(), {
      method: request.method,
      headers: requestHeaders,
      body: ["GET", "HEAD"].includes(request.method) ? undefined : request.body,
      redirect: "manual",
    });
    const response = await fetch(upstreamRequest);
    const headers = new Headers(response.headers);
    headers.set("X-Kakeya-Surface", "inference-network");
    headers.set("X-Content-Type-Options", "nosniff");
    headers.set("Referrer-Policy", "same-origin");
    if (path.startsWith("/v1/proof/")) {
      headers.set("Cache-Control", "private, no-store, no-cache, must-revalidate, max-age=0");
      headers.set("CDN-Cache-Control", "no-store");
      headers.set("Cloudflare-CDN-Cache-Control", "no-store");
      headers.set("Surrogate-Control", "no-store");
      headers.set("Pragma", "no-cache");
      headers.set("Expires", "0");
    }

    return new Response(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers,
    });
  },
};
