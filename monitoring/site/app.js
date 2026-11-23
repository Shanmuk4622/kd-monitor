"use strict";
const el = id => document.getElementById(id);
function node(tag, value, cls) {
  const n = document.createElement(tag);
  if (value != null) n.textContent = value;
  if (cls) n.className = cls;
  return n;
}
const fmt = (v, digits = 0, unit = "") => Number.isFinite(v) ? v.toFixed(digits) + unit : "—";
const localTime = t => new Date(t).toLocaleString();
let workers = [], events = [];
function render() {
  const cards = [];
  for (let id = 0; id < 4; id++) {
    const w = workers.find(x => x.worker_id === id), s = w?.sample || {};
    const age = w ? Math.max(0, Math.floor((Date.now() - Date.parse(w.received_at)) / 1000)) : null;
    const state = !w ? "No report" : s.status === "running" && age > 90 ? "stale" : s.status;
    const box = node("article", null, "card"), head = node("div", null, "card-top");
    head.append(node("h3", "Worker " + id), node("span", state, "state " + state));
    box.append(head, node("p", s.run_id || "Awaiting worker", "run"));
    const dl = node("dl");
    for (const [name, value] of [["Model",s.model ?? "—"],["Epoch",s.epoch ?? "—"],["CPU",fmt(s.cpu,1,"%")],
      ["Batch loss",fmt(s.loss,4)],["Batch accuracy",fmt(s.accuracy,1,"%")]]) {
      const cell = node("div"); cell.append(node("dt",name),node("dd",value)); dl.append(cell);
    }
    box.append(dl);
    const gpus = s.gpus || [];
    if (!gpus.length) box.append(node("p","GPU telemetry unavailable","gpu"));
    for (const g of gpus) box.append(node("p", "GPU " + g.index + " · " + fmt(g.utilization,0,"%") +
      " · " + fmt(g.used_mib) + " / " + fmt(g.total_mib) + " MiB","gpu"));
    box.append(node("p",s.comment || "Waiting for the first update."),
      node("small",w ? "Last received " + localTime(w.received_at) : "No update received"));
    cards.push(box);
  }
  el("workers").replaceChildren(...cards);
  const rows = events.map(e => {
    const tr = node("tr");
    for (const v of [localTime(e.recorded_at),e.worker_id,e.run_id,e.comment]) tr.append(node("td",v));
    return tr;
  });
  if (!rows.length) {
    const tr = node("tr"), td = node("td","No events received yet."); td.colSpan=4; tr.append(td); rows.push(tr);
  }
  el("events").replaceChildren(...rows);
}
let client, channel, version=0, viewer=null, request=0, timer, fallback;
async function refresh() {
  if (!viewer) return;
  const ticket=++request, v=version;
  try {
    const [w,e] = await Promise.all([client.from("kd_workers").select("*"),
      client.from("kd_events").select("*").order("id",{ascending:false}).limit(40)]);
    if (v!==version || ticket!==request) return;
    if (w.error || e.error) throw w.error || e.error;
    workers=w.data; events=e.data; el("notice").textContent=""; render();
  } catch (err) {
    if (v===version && ticket===request) el("notice").textContent="Could not refresh. " + err.message;
  }
}
function schedule() { clearTimeout(timer); timer=setTimeout(refresh,150); }
function showSession(session) {
  const user=session?.user, allowed=user?.app_metadata?.kd_monitor==="viewer";
  const id=allowed ? user.id : null;
  el("login").hidden=Boolean(user); el("logout").hidden=!user; el("dashboard").hidden=!allowed;
  if (!allowed) {
    el("connection").textContent=user ? "Access required" : "Signed out";
    el("notice").textContent=user ? "Ask the project owner to grant this account viewer access, then sign in again." : "";
  }
  if (viewer===id) return;
  version++; viewer=id; clearTimeout(timer); clearInterval(fallback);
  if (channel) {client.removeChannel(channel); channel=null;}
  workers=[]; events=[]; render();
  if (!id) return;
  const v=version;
  el("connection").textContent="Connecting"; refresh();
  channel=client.channel("kd-dashboard-" + v)
    .on("postgres_changes",{event:"*",schema:"public",table:"kd_workers"},schedule)
    .on("postgres_changes",{event:"INSERT",schema:"public",table:"kd_events"},schedule)
    .subscribe(status=>{
      if(v!==version) return;
      el("connection").textContent=status==="SUBSCRIBED" ? "Live" : "Reconnecting · periodic refresh";
      schedule();
    });
  fallback=setInterval(refresh,60000);
}
function init() {
  const config=window.KD_CONFIG;
  if (!config?.url || !config.key || !window.supabase) {
    el("connection").textContent="Setup required";
    el("notice").textContent="Set the Supabase URL and public key in config.js and allow the dashboard SDK to load.";
    return;
  }
  client=window.supabase.createClient(config.url,config.key);
  client.auth.onAuthStateChange((_event,session)=>setTimeout(()=>showSession(session),0));
  el("login").addEventListener("submit",async e=>{
    e.preventDefault();
    const {error}=await client.auth.signInWithPassword({email:el("email").value.trim(),password:el("password").value});
    el("notice").textContent=error ? error.message : ""; if (!error) el("password").value="";
  });
  el("logout").onclick=async()=>{const {error}=await client.auth.signOut(); if(error) el("notice").textContent=error.message;};
  setInterval(()=>{if(viewer) render();},10000);
  window.addEventListener("online",schedule);
}
init();
