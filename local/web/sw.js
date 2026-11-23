// Makes Wadia installable as an app. Wadia needs the laptop to answer, so nothing is cached:
// requests go straight to the network, with a friendly page when the laptop can't be reached.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));
self.addEventListener("fetch", (event) => {
  if (event.request.mode !== "navigate") return;
  event.respondWith(
    fetch(event.request).catch(() => new Response(
      `<!doctype html><meta name="viewport" content="width=device-width,initial-scale=1">
       <body style="background:#0b0b14;color:#f2f2f7;font-family:sans-serif;display:grid;place-items:center;height:100vh;margin:0;text-align:center;padding:24px">
       <div><h2>Can't reach Wadia</h2><p style="color:#9a9ab0">Make sure the laptop is on, the Wadia app is open,<br>and its Public link is on.</p>
       <button onclick="location.reload()" style="background:#8b7bff;color:#fff;border:0;border-radius:12px;padding:12px 20px;font-size:16px">Try again</button></div>`,
      { headers: { "Content-Type": "text/html" } })),
  );
});
