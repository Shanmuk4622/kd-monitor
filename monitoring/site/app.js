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
let trends = [], activity = [], memory = [], historyReady = false, nextHistory = 0;
const plans = window.KD_PLAN || [];
function duration(seconds) {
  if (!Number.isFinite(seconds)) return "—";
  const minutes = Math.floor(Math.max(seconds,0) / 60);
  return minutes >= 60 ? Math.floor(minutes/60)+"h "+minutes%60+"m" : minutes+"m";
}
function runState(plan, w) {
  const s=w?.sample, d=s?.details;
  if (d?.audited?.includes(plan.id)) return "Audited";
  if (s?.run_id===plan.id && s.status==="running")
    return Date.now()-Date.parse(w.received_at)>90000 ? "Last reported" : "Active";
  if (d?.resumable?.includes(plan.id)) return "Resumable";
  return Array.isArray(d?.audited) ? "Pending" : "Unconfirmed";
}
function chart(title, key, points, percent) {
  const figure=node("figure");
  figure.append(node("figcaption",title));
  const values=points.filter(p=>Number.isFinite(p[key]));
  if (!values.length) {figure.append(node("small","Awaiting samples"));return figure;}
  const ns="http://www.w3.org/2000/svg", svg=document.createElementNS(ns,"svg");
  svg.setAttribute("viewBox","0 0 240 75"); svg.setAttribute("role","img");
  svg.setAttribute("aria-label",title+" over time");
  const hi=percent?100:Math.max(0.01,...values.map(p=>p[key]));
  const start=Date.parse(points[0].sampled_at), end=Date.parse(points.at(-1).sampled_at);
  let segment=[], previous=null;
  function finish() {
    if (!segment.length) return;
    const line=document.createElementNS(ns,segment.length===1?"circle":"polyline");
    if (segment.length===1) {
      line.setAttribute("cx",segment[0][0]);line.setAttribute("cy",segment[0][1]);line.setAttribute("r","2");
      line.setAttribute("fill","#20745d");
    } else {line.setAttribute("points",segment.join(" "));line.setAttribute("fill","none");}
    line.setAttribute("stroke","#20745d");line.setAttribute("stroke-width","2");svg.append(line);segment=[];
  }
  for (const p of points) {
    const t=Date.parse(p.sampled_at);
    if (!Number.isFinite(p[key]) || (previous && (p.run_id!==previous.run_id ||
        t-Date.parse(previous.sampled_at)>600000))) finish();
    if (Number.isFinite(p[key])) segment.push([3+234*(t-start)/Math.max(1,end-start),71-66*p[key]/hi]);
    previous=p;
  }
  finish();figure.append(svg,node("small","Scale 0–"+fmt(hi,percent?0:2,percent?"%":"")+
    " · Latest "+fmt(values.at(-1)[key],percent?1:3,percent?"%":"")));
  return figure;
}
function renderDetail() {
  const id=Number(el("focus").value), w=workers.find(x=>x.worker_id===id);
  const items=plans.filter(p=>p.worker===id), filter=el("filter").value;
  const details=w?.sample?.details;
  el("plan-note").textContent=Array.isArray(details?.selected)
    ? details.selected.length+" selected this session. Audited fits have passed the Hugging Face evidence check."
    : "Assignment is known. Completion is unconfirmed until this worker sends an audit inventory.";
  const rows=[];
  for (const p of items) {
    const state=runState(p,w);
    if (filter && state!==filter) continue;
    const a=activity.find(x=>x.worker_id===id && x.run_id===p.id);
    const tr=node("tr");
    for (const value of [p.dataset,p.model,p.arm,p.seed,p.epochs,state,duration(a?.observed_seconds)])
      tr.append(node("td",value));
    rows.push(tr);
  }
  if (!rows.length) {
    const tr=node("tr"),td=node("td","No matching runs.");td.colSpan=7;tr.append(td);rows.push(tr);
  }
  el("inventory").replaceChildren(...rows);
  const points=(historyReady?trends:memory).filter(p=>p.worker_id===id);
  el("history-note").textContent=(historyReady ? "Five-minute samples · last 12 hours" : "Live samples collected in this tab")+
    (points.length?" · "+localTime(points[0].sampled_at)+" – "+localTime(points.at(-1).sampled_at):"");
  el("charts").replaceChildren(chart("Batch loss","loss",points,false),chart("Batch accuracy","accuracy",points,true),
    chart("GPU 0 utilization","gpu",points,true),chart("CPU utilization","cpu",points,true));
}
function render() {
  const cards = [];
  for (let id = 0; id < 4; id++) {
    const w = workers.find(x => x.worker_id === id), s = w?.sample || {};
    const age = w ? Math.max(0, Math.floor((Date.now() - Date.parse(w.received_at)) / 1000)) : null;
    const state = !w ? "No report" : s.status === "running" && age > 90 ? "stale" : s.status;
    const box = node("article", null, "card"), head = node("div", null, "card-top");
    head.append(node("h3", "Worker " + id), node("span", state, "state " + state));
    box.append(head, node("p", s.run_id || "Awaiting worker", "run"));
    const owned=plans.filter(p=>p.worker===id), d=s.details || {};
    const plan=owned.find(p=>p.id===s.run_id);
    const progress=Number.isFinite(d.update) ? d.update : Number(/Local checkpoint at update (\d+)/.exec(s.comment||"")?.[1]);
    const activeEnd=state==="running" ? Date.now() : Date.parse(s.sampled_at);
    const elapsed=(activeEnd-Date.parse(s.started_at))/1000;
    const observed=activity.find(a=>a.worker_id===id && a.run_id===s.run_id);
    const done=owned.filter(p=>runState(p,w)==="Audited").length;
    box.append(node("p",owned.length+" assigned · "+(Array.isArray(d.audited)
      ? done+" audited · "+(owned.length-done)+" unfinished" : "awaiting audit inventory")));
    if (plan && Number.isFinite(progress)) {
      const bar=node("progress");bar.max=plan.updates;bar.value=progress;bar.setAttribute("aria-label","Fit updates");
      box.append(bar,node("small",progress.toLocaleString()+" / "+plan.updates.toLocaleString()+" updates"));
    }
    const dl = node("dl");
    for (const [name, value] of [["Model",s.model ?? "—"],["Epoch",s.epoch ?? "—"],["CPU",fmt(s.cpu,1,"%")],
      ["Batch loss",fmt(s.loss,4)],["Batch accuracy",fmt(s.accuracy,1,"%")],
      ["Session elapsed",duration(elapsed)],["Current run elapsed",duration(activeEnd/1000-d.fit_started_at)],
      ["Fit observed",duration(observed?.observed_seconds)],
      ["Validation",fmt(d.validation_accuracy,1,"%")+(d.validation_epoch?" (epoch "+d.validation_epoch+")":"")],
      ["Learning rate",fmt(d.learning_rate,6)]]) {
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
  el("summary").textContent=plans.length+" scheduled fits across four notebooks";
  renderDetail();
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
    workers=w.data; events=e.data;
    for (const row of workers) {
      const s=row.sample;
      if (!memory.some(p=>p.worker_id===row.worker_id && p.sampled_at===s.sampled_at))
        memory.push({worker_id:row.worker_id,sampled_at:s.sampled_at,run_id:s.run_id,loss:s.loss,
          accuracy:s.accuracy,cpu:s.cpu,gpu:s.gpus?.[0]?.utilization});
    }
    memory=memory.slice(-600).sort((a,b)=>Date.parse(a.sampled_at)-Date.parse(b.sampled_at));
    el("notice").textContent=""; render();
    if (Date.now()>=nextHistory) {
      nextHistory=Date.now()+60000;
      const [t,a]=await Promise.all([client.from("kd_trends").select("*")
          .gte("bucket",new Date(Date.now()-12*3600000).toISOString()).order("sampled_at"),
        client.from("kd_run_activity").select("*")]);
      if (v!==version) return;
      historyReady=!t.error;
      if (!t.error) trends=t.data;
      if (!a.error) activity=a.data;
      render();
    }
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
  workers=[]; events=[]; trends=[]; activity=[]; memory=[]; historyReady=false;nextHistory=0;render();
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
  el("focus").onchange=renderDetail;el("filter").onchange=renderDetail;
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
