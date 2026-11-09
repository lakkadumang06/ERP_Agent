/* ERP-Agent web UI: a thin client over the FastAPI endpoints. No build step. */
"use strict";

const $ = (id) => document.getElementById(id);
const fmt = (n) => Number(n).toLocaleString("en-GB", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const isNarrow = () => window.matchMedia("(max-width: 1079px)").matches;

const SAMPLES = [
  ["po_001.txt", "PO · Shree Ganesh Jewels", "clean"], ["po_002.txt", "Bestellung · Rhein Textil", "clean"],
  ["po_003.txt", "PO · Lotus Boutique", "clean"], ["po_004.txt", "Bestellung · Schwaben Stoffe", "clean"],
  ["inv_001.txt", "Invoice · Spice Bite", "clean"], ["inv_002.txt", "Rechnung · Bayerische Feinkost", "clean"],
  ["po_001.pdf", "PO · Shree Ganesh (PDF)", "clean"],
  ["err_wrong_total.txt", "Wrong total", "error"], ["err_unknown_sku.txt", "Unknown SKU", "error"],
  ["err_over_stock.txt", "Quantity above stock", "error"], ["err_unknown_customer.txt", "Unknown customer", "error"],
  ["err_wrong_total.pdf", "Wrong total (PDF)", "error"],
];
const NODES = ["load_document", "extract", "rule_check", "erp_check", "human_review", "save_order"];
const STATUS = {
  saved: ["✅", "Order saved in the ERP and stock reduced."],
  needs_review: ["🧑‍⚖️", "The agent found issues. A human decides."],
  needs_re_review: ["🔄", "The ERP changed after you looked. Check the new values, then decide again."],
  rejected: ["⛔", "Rejected by the reviewer. Nothing was saved."],
  error: ["⚠️", "Processing failed. See the issues below."],
};

const REVIEW_STATES = new Set(["needs_review", "needs_re_review"]);

let activeTab = "upload", selectedFile = null, selectedSample = null, currentThread = null, currentRecheck = null, timer = null, toastTimer = null;

function toast(message, kind = "") {
  const el = $("toast");
  el.textContent = message; el.className = `toast ${kind}`; el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.hidden = true), kind === "error" ? 6000 : 3000);
}

// ---------- tabs ----------
document.querySelectorAll(".tab").forEach((btn) =>
  btn.addEventListener("click", () => {
    activeTab = btn.dataset.tab;
    document.querySelectorAll(".tab").forEach((b) => { const on = b === btn; b.classList.toggle("active", on); b.setAttribute("aria-selected", on); });
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${activeTab}`));
  })
);

// ---------- upload ----------
const dropzone = $("dropzone"), fileInput = $("file-input");
fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
["dragenter", "dragover"].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); }));
dropzone.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));
function setFile(file) {
  selectedFile = file || null;
  $("file-name").textContent = file ? `${file.name} · ${Math.max(1, Math.round(file.size / 1024))} KB` : "";
}

// ---------- samples ----------
const sampleList = $("sample-list");
SAMPLES.forEach(([name, label, kind]) => {
  const b = document.createElement("button");
  b.className = "sample"; b.type = "button"; b.setAttribute("role", "option");
  b.innerHTML = `<span class="tag ${kind}">${kind}</span><span>${esc(label)}</span><small>${esc(name)}</small>`;
  b.addEventListener("click", () => {
    selectedSample = name;
    sampleList.querySelectorAll(".sample").forEach((x) => x.classList.toggle("selected", x === b));
  });
  sampleList.appendChild(b);
});

// ---------- submit / review ----------
$("submit").addEventListener("click", async () => {
  const provider = $("provider").value;
  try {
    let request;
    if (activeTab === "paste") {
      const text = $("text-input").value.trim();
      if (!text) throw new Error("Paste some document text first.");
      request = () => fetch("/documents", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text, provider: provider || null }) });
    } else {
      let file = selectedFile;
      if (activeTab === "sample") {
        if (!selectedSample) throw new Error("Choose a sample document.");
        const blob = await (await fetch(`/samples/${selectedSample}`)).blob();
        file = new File([blob], selectedSample);
      }
      if (!file) throw new Error("Choose a file first.");
      const form = new FormData();
      form.append("file", file);
      if (provider) form.append("provider", provider);
      request = () => fetch("/documents", { method: "POST", body: form });
    }
    await runRequest(request, "Processing document", true);
  } catch (e) { toast(e.message, "error"); }
});
$("approve").addEventListener("click", () => review(true));
$("reject").addEventListener("click", () => review(false));
$("recheck").addEventListener("click", recheckNow);
async function review(approve) {
  if (!currentThread) return;
  // Approve against the version on screen: the server refuses if the ERP drifted further.
  const seen_drift_hash = currentRecheck ? currentRecheck.drift_hash : null;
  await runRequest(
    () => fetch(`/documents/${currentThread}/review`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ approve, seen_drift_hash }) }),
    approve ? "Re-checking ERP & saving" : "Rejecting", false
  );
}
async function recheckNow() {
  if (!currentThread) return;
  try {
    const res = await fetch(`/documents/${currentThread}`);
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || `HTTP ${res.status}`);
    render(body);
    const n = body.recheck ? body.recheck.drift.length : 0;
    toast(n ? `${n} change(s) in the ERP since the proposal.` : "No change in the ERP since the proposal.", n ? "" : "ok");
  } catch (e) { toast(e.message, "error"); }
}

async function runRequest(makeRequest, busyText, fullPipeline) {
  setBusy(true, busyText, fullPipeline);
  const started = Date.now();
  try {
    const res = await makeRequest();
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || `HTTP ${res.status}`);
    render(body);
    $("elapsed").textContent = `${((Date.now() - started) / 1000).toFixed(1)} s`;
    const [, text] = STATUS[body.status] || ["", body.status];
    toast(text, body.status === "saved" ? "ok" : body.status === "needs_review" ? "" : "error");
    await loadErp();
  } catch (e) {
    toast(e.message, "error");
    $("result-empty").hidden = false;
    setPipeline([], "error");
  } finally { setBusy(false); }
}

function setBusy(busy, text, fullPipeline) {
  ["submit", "approve", "reject", "recheck"].forEach((id) => ($(id).disabled = busy));
  clearInterval(timer);
  if (!busy) { $("result-busy").hidden = true; return; }
  $("result-empty").hidden = true; $("result").hidden = true; $("result-busy").hidden = false;
  $("busy-text").textContent = `${text}…`; $("elapsed").textContent = "";
  const t0 = Date.now();
  // The request is one round trip, so the tracker is time-based while it runs
  // and replaced by the real node log when the response arrives.
  const guess = fullPipeline ? ["load_document", "extract", "rule_check", "erp_check"] : ["human_review", "save_order"];
  let idx = 0;
  setPipeline([], null); markPipeline(guess[0], "active");
  timer = setInterval(() => {
    const s = (Date.now() - t0) / 1000;
    $("busy-text").textContent = `${text}… ${s.toFixed(0)} s`;
    const next = fullPipeline ? (s < 3 ? 0 : s < 40 ? 1 : s < 42 ? 2 : 3) : (s < 1 ? 0 : 1);
    if (next !== idx) { markPipeline(guess[idx], "done"); idx = next; markPipeline(guess[idx], "active"); }
  }, 500);
  if (isNarrow()) $("result-card").scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---------- pipeline tracker ----------
function markPipeline(node, cls) {
  const li = document.querySelector(`#pipeline li[data-node="${node}"]`);
  if (li) li.className = cls;
}
function setPipeline(log, status) {
  NODES.forEach((n) => markPipeline(n, ""));
  const ran = new Set(log.map((entry) => entry.split(":")[0]));
  log.forEach((entry) => {
    const [node, rest = ""] = entry.split(/:(.*)/s);
    const failed = /failed|skipped/.test(rest);
    const issues = /(\d+) issue\(s\)/.exec(rest);
    let cls = failed ? (rest.includes("skipped") ? "skip" : "fail") : "done";
    if (issues && Number(issues[1]) > 0) cls = "warn";
    if (node === "human_review") cls = rest.includes("rejected") ? "fail" : rest.includes("refused") ? "warn" : "done";
    markPipeline(node, cls);
  });
  if (status === "needs_review" && !ran.has("human_review")) markPipeline("human_review", "active");
  if (status === "needs_re_review") markPipeline("human_review", "active");
  if (REVIEW_STATES.has(status)) markPipeline("save_order", "");
}

// ---------- render ----------
function render(r) {
  currentThread = REVIEW_STATES.has(r.status) ? r.thread_id : null;
  currentRecheck = r.recheck || null;
  $("result-empty").hidden = true; $("result-busy").hidden = true; $("result").hidden = false;
  const [icon, text] = STATUS[r.status] || ["•", r.status];
  $("banner").className = `banner ${r.status}`;
  $("status-icon").textContent = icon;
  $("status-pill").textContent = r.status.replaceAll("_", " ");
  $("status-text").textContent = r.order_id ? `${text} Order #${r.order_id}.` : text;

  const d = r.document;
  $("document-block").hidden = !d;
  if (d) {
    $("doc-number").textContent = d.document_number;
    $("doc-type").textContent = d.doc_type === "invoice" ? "Invoice / Rechnung" : "Purchase order / Bestellung";
    $("doc-customer").textContent = d.customer_name;
    $("doc-date").textContent = d.document_date || "–";
    $("doc-lines").innerHTML = d.lines.map((l) =>
      `<tr><td><code>${esc(l.sku || "?")}</code></td><td>${esc(l.description)}</td><td class="num">${l.quantity}</td><td class="num">${fmt(l.unit_price)}</td><td class="num">${fmt(l.line_total)}</td></tr>`).join("");
    $("doc-total").textContent = `${fmt(d.stated_total)} ${d.currency}`;
  }
  $("issues-block").hidden = !r.issues.length;
  $("issues").innerHTML = r.issues.map((i) => `<li class="${i.severity}"><code>${esc(i.code)}</code><span>${esc(i.message)}</span></li>`).join("");
  renderRecheck(r);
  $("result-log").innerHTML = (r.log || []).map((l) => `<li>${esc(l)}</li>`).join("");
  $("raw-json").textContent = JSON.stringify(r, null, 2);
  setPipeline(r.log || [], r.status);
  if (isNarrow()) $("result-card").scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---------- approval re-check: "at proposal" vs "now" ----------
function renderRecheck(r) {
  const rc = r.recheck, inReview = REVIEW_STATES.has(r.status);
  $("recheck-block").hidden = !rc;
  $("review-block").hidden = !inReview;
  $("recheck").hidden = !inReview;
  if (!rc) return;
  const d = r.document, drift = rc.drift || [];
  const changed = new Set(drift.map((x) => `${x.field}:${x.sku || ""}`));
  const cell = (val, key, cls = "num") => `<td class="${cls}${changed.has(key) ? " changed" : ""}">${val}</td>`;
  const price = (l) => (l && l.found ? fmt(l.unit_price_catalogue) : "–");
  const stock = (l) => (l && l.found ? l.stock : "–");
  const qty = {};
  (d ? d.lines : []).forEach((l) => { if (l.sku) qty[l.sku] = (qty[l.sku] || 0) + l.quantity; });
  const rows = rc.snapshot.lines.map((then) => {
    const now = rc.current.lines.find((x) => x.sku === then.sku);
    const missing = then.found && now && !now.found;
    return `<tr><td><code>${esc(then.sku)}</code>${missing ? ' <span class="tag error">removed</span>' : ""}</td><td class="num">${qty[then.sku] ?? ""}</td>`
      + cell(price(then), "", "num then") + cell(price(now), `price:${then.sku}`, "num now")
      + cell(stock(then), "", "num then") + cell(stock(now), `stock:${then.sku}`, "num now") + "</tr>";
  });
  const cName = (c) => (c && c.found ? esc(c.name) : '<span class="tag error">not found</span>');
  rows.unshift(`<tr><td>Customer</td><td></td>${cell(cName(rc.snapshot.customer), "", "then")}${cell(cName(rc.current.customer), "customer:", "now")}<td class="then"></td><td class="now"></td></tr>`);
  $("recheck-rows").innerHTML = rows.join("");
  $("recheck-times").textContent = `snapshot ${rc.snapshot_at} · checked ${rc.checked_at} · drift hash ${rc.drift_hash}`;

  const banner = $("drift-banner");
  banner.hidden = !drift.length;
  banner.className = `drift-banner ${rc.blocked ? "block" : "warn"}`;
  $("drift-icon").textContent = rc.blocked ? "⛔" : "⚠️";
  $("drift-title").textContent = rc.blocked
    ? "The ERP changed since the proposal and the order can no longer be saved."
    : `${drift.length} change(s) in the ERP since the proposal. Check the highlighted cells before approving.`;
  $("drift-list").innerHTML = drift.map((x) => `<li class="${x.severity}"><code>${esc(x.severity)}</code><span>${esc(x.message)}</span></li>`).join("");

  $("approve").disabled = rc.blocked;
  $("approve").title = rc.blocked ? "A blocking change prevents saving" : "";
  $("review-title").textContent = r.status === "needs_re_review" ? "Look again, then decide" : "Human review required";
  $("review-hint").textContent = rc.blocked
    ? "A blocking change (stock, customer or SKU) means this order cannot be saved as it stands. Re-check later or reject it."
    : r.status === "needs_re_review"
      ? "Your approval was not applied because the ERP changed after you looked. The table shows today's values; approve again if they are still acceptable."
      : "Approve to save the order anyway, or reject it. Approval still cannot bypass stock or unknown SKUs, and it only counts for the ERP values shown above.";
}

// ---------- ERP state ----------
async function loadErp() {
  try {
    const [orders, products] = await Promise.all([fetch("/orders").then((r) => r.json()), fetch("/products").then((r) => r.json())]);
    $("orders").innerHTML = orders.length
      ? orders.map((o) => `<tr><td>${o.id}</td><td>${esc(o.customer_name)}</td><td class="mono">${esc(o.document_number || "–")}</td><td>${o.lines.map((l) => `<span class="lines-chip">${esc(l.sku)} ×${l.quantity}</span>`).join("")}</td><td class="num">${fmt(o.total)}</td></tr>`).join("")
      : `<tr><td colspan="5" class="hint">No orders yet. Process a clean document to create one.</td></tr>`;
    const maxStock = Math.max(...products.map((p) => p.stock), 1);
    const low = products.filter((p) => p.stock < 20).length;
    $("products").innerHTML = products.map((p) => {
      const level = p.stock < 20 ? "low" : p.stock < 60 ? "mid" : "";
      const pct = Math.max(4, Math.round((p.stock / maxStock) * 100));
      return `<tr><td><code>${esc(p.sku)}</code></td><td>${esc(p.name)}<span class="cat ${esc(p.category)}">${esc(p.category)}</span></td><td class="num">${fmt(p.unit_price)}</td><td class="num"><span class="stock ${level}"><span class="bar"><i style="width:${pct}%"></i></span><b>${p.stock}</b></span></td></tr>`;
    }).join("");
    $("stat-orders").textContent = orders.length; $("stat-products").textContent = products.length; $("stat-low").textContent = low;
  } catch { toast("Could not load ERP data. Is the API running?", "error"); }
}
$("refresh").addEventListener("click", loadErp);

// ---------- providers dropdown ----------
async function loadProviders() {
  try {
    const providers = await fetch("/providers").then((r) => r.json());
    const sel = $("provider");
    sel.innerHTML = "";
    providers.filter((p) => p.configured).forEach((p) => {
      const opt = document.createElement("option");
      opt.value = p.name;
      opt.textContent = `${p.label} · ${p.model}${p.default ? " (default)" : ""}`;
      if (p.default) opt.selected = true;
      sel.appendChild(opt);
    });
    if (!sel.options.length) {
      sel.innerHTML = `<option value="">No provider configured – add a key to .env</option>`;
    }
  } catch { /* keep the static default option */ }
}
loadProviders();

// ---------- health ----------
fetch("/health").then((r) => r.json()).then((h) => {
  const chip = $("health");
  chip.className = "status-chip ok"; chip.innerHTML = `<i></i>online · ${esc(h.model || h.provider || "")}`;
}).catch(() => { $("health").className = "status-chip bad"; $("health").innerHTML = "<i></i>API offline"; });

loadErp();
