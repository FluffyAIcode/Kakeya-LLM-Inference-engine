const ORIGIN = "https://agent.kakeya.ai";

export default {
  async fetch(request) {
    const incoming = new URL(request.url);
    let path = incoming.pathname;

    if (path === "/" || path === "/index.html") {
      path = "/network";
    } else if (path === "/healthz") {
      path = "/v1/network/summary";
    } else if (
      path !== "/network" &&
      !path.startsWith("/v1/network/")
    ) {
      path = "/network";
    }

    const target = new URL(path + incoming.search, ORIGIN);
    const upstreamRequest = new Request(target, request);
    const response = await fetch(upstreamRequest);
    const headers = new Headers(response.headers);
    headers.set("X-Kakeya-Surface", "inference-network");
    headers.set("X-Content-Type-Options", "nosniff");
    headers.set("Referrer-Policy", "same-origin");

    return new Response(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers,
    });
  },
};
