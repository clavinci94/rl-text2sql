"""Prototype: self-service data questions for a business department, running fully locally.

A small web app on top of reliable.py. Staff ask in German (or French / Italian), the model
writes SQL, the database runs it read-only, and the app shows the result together with the
SQL and how sure the system is. Below the confidence threshold it says so instead of
pretending. Every question is logged (audit log); confirmed or corrected queries are stored
as new training examples for the next RL round.

    python app.py --adapter checkpoints/grpo_local_binary_lr1e-6_g8_kl      # Mac, MLX
    python app.py --backend hf --model checkpoints/cloud_.../merged         # Linux / CUDA
    then open http://127.0.0.1:8000

Only the Python standard library plus the model backend. Database: data/versicherung.sqlite
(python insurance_db.py build). Logs: logs/audit.jsonl, logs/feedback.jsonl.
"""
import argparse
import getpass
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from insurance_db import CATALOG, DB_ID, GLOSSARY, load_questions, prompt_schema
from reliable import SYSTEM_PROMPT, answer, make_generator, run_candidate
from sql_env import ROOT, execute_sql_full

LOG_DIR = ROOT / "logs"
STATE = {}


def log(name, record):
    LOG_DIR.mkdir(exist_ok=True)
    with open(LOG_DIR / name, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def meta():
    import sqlite3
    from sql_env import db_path
    con = sqlite3.connect(f"file:{db_path(DB_ID)}?mode=ro", uri=True)
    tables = []
    for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY rowid"):
        desc, cols = CATALOG.get(t, ("", {}))
        n = con.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0]
        tables.append({"name": t, "desc": desc, "rows": n,
                       "columns": [{"name": c[1], "desc": cols.get(c[1], "")} for c in con.execute(f'PRAGMA table_info("{t}")')]})
    con.close()
    examples = [q["question"] for q in load_questions() if q["lang"] == "de"][::5][:8]
    return {"db": DB_ID, "tables": tables, "glossary": GLOSSARY, "examples": examples,
            "threshold": STATE["threshold"], "user": STATE["user"], "model": STATE["model_label"]}


def ask(question):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Database schema:\n{STATE['schema']}\n\nQuestion: {question}"}]
    with STATE["lock"]:  # one generation at a time
        res = answer(STATE["gen"], msgs, DB_ID, STATE["samples"], STATE["retries"])
    c = res["chosen"]
    sure = res["confidence"] >= STATE["threshold"] and c["key"] is not None
    out = {"id": uuid.uuid4().hex[:10], "question": question, "sql": c["sql"], "error": c["error"],
           "degenerate": c.get("degenerate", False), "columns": c.get("columns"), "rows": c.get("rows"),
           "n_rows": c.get("n_rows"), "confidence": res["confidence"], "n_voters": res["n_voters"],
           "votes": round(res["confidence"] * res["n_voters"]), "decision": "answer" if sure else "unsure",
           "alternatives": res["alternatives"][:3], "retried": len(res["chain"]) > 1,
           "latency_s": res["latency_s"]["with_vote"]}
    log("audit.jsonl", {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "user": STATE["user"], "id": out["id"],
                        "question": question, "sql": c["sql"], "confidence": res["confidence"],
                        "decision": out["decision"], "n_rows": out["n_rows"], "latency_s": out["latency_s"]})
    return out


def run_sql(sql):
    """Run a query a person wrote or corrected (read-only, same sandbox as the model)."""
    c = run_candidate(DB_ID, f"```sql\n{sql}\n```")
    return {"sql": c["sql"], "error": c["error"], "columns": c.get("columns"), "rows": c.get("rows"),
            "n_rows": c.get("n_rows"), "degenerate": c.get("degenerate", False)}


def feedback(data):
    """Confirmed or corrected query -> logs/feedback.jsonl. Each confirmed pair (question, SQL)
    is a verified training example with an execution reward, without a hand-written gold set."""
    rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "user": STATE["user"], "id": data.get("id"),
           "verdict": data.get("verdict"), "question": data.get("question"), "sql": data.get("sql"),
           "model_sql": data.get("model_sql"), "db_id": DB_ID}
    if rec["verdict"] in ("confirmed", "corrected"):
        _rows, _cols, err = execute_sql_full(DB_ID, rec["sql"] or "")
        if err:
            return {"ok": False, "error": f"Die Abfrage läuft nicht: {err}"}
    log("feedback.jsonl", rec)
    n = sum(1 for _ in open(LOG_DIR / "feedback.jsonl", encoding="utf-8"))
    return {"ok": True, "stored": n}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body.encode() if isinstance(body, str) else json.dumps(body, ensure_ascii=False, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/":
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/api/meta":
            self._send(200, meta())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path == "/api/ask":
                q = (data.get("question") or "").strip()
                self._send(200, ask(q) if q else {"error": "Bitte eine Frage eingeben."})
            elif self.path == "/api/run":
                self._send(200, run_sql(data.get("sql", "")))
            elif self.path == "/api/feedback":
                self._send(200, feedback(data))
            else:
                self._send(404, {"error": "not found"})
        except Exception as e:  # show the problem in the page instead of a dead request
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, *_):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mlx", choices=["mlx", "hf"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=0.5, help="min. share of agreeing answers to answer")
    ap.add_argument("--no-catalog", action="store_true", help="prompt without column descriptions and glossary")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--user", default=getpass.getuser(), help="name written to the audit log")
    args = ap.parse_args()
    print("loading model ...")
    STATE.update(gen=make_generator(args.backend, args.model, args.adapter), lock=threading.Lock(),
                 schema=prompt_schema(catalog=not args.no_catalog), samples=args.samples, retries=args.retries,
                 threshold=args.threshold, user=args.user,
                 model_label=args.model.split("/")[-1] + (" + RL-Adapter" if args.adapter else ""))
    print(f"ready: http://127.0.0.1:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


PAGE = r"""<!doctype html>
<html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Datenabfrage</title>
<style>
:root{--bg:#f3f5f7;--panel:#fff;--line:#d9e0e7;--fg:#13202b;--fg2:#4b5a67;--mut:#7a8794;--acc:#2a78d6;--accs:#e3eefb;
--ok:#0f8a3c;--oks:#e3f4e9;--warn:#a86a00;--warns:#fdf2dc;--bad:#c23a32;--bads:#fbe7e5;--code:#f0f3f6;
font-family:-apple-system,"Segoe UI",system-ui,sans-serif;color-scheme:light}
@media (prefers-color-scheme:dark){:root{--bg:#0e151c;--panel:#151e27;--line:#2a3743;--fg:#e5ecf2;--fg2:#a7b4c0;--mut:#7f8d9a;
--acc:#4a90e8;--accs:#17304d;--ok:#3fbf6a;--oks:#14301f;--warn:#e6a83a;--warns:#33280f;--bad:#ec6a62;--bads:#3a1a18;--code:#0b1117;color-scheme:dark}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font-size:15px;line-height:1.5}
header{display:flex;align-items:center;gap:14px;padding:12px 20px;background:var(--panel);border-bottom:1px solid var(--line)}
header b{font-size:17px}.pill{font-size:12px;border:1px solid var(--line);border-radius:99px;padding:3px 10px;color:var(--fg2)}
.pill i{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--ok);margin-right:6px}
.sp{flex:1}main{display:grid;grid-template-columns:300px 1fr;gap:20px;padding:20px;max-width:1400px;margin:0 auto}
@media(max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px}
.side{display:grid;gap:16px;align-content:start}.side h3{margin:0 0 8px;font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut)}
details{border-top:1px solid var(--line);padding:6px 0}details summary{cursor:pointer;font-family:ui-monospace,Menlo,monospace;font-size:13px}
details .d{font-size:12px;color:var(--mut)}.col{font-size:12px;padding:2px 0 2px 12px}.col code{font-size:12px}.col span{color:var(--mut)}
.gl{font-size:12.5px;padding:4px 0;border-top:1px solid var(--line)}.gl b{display:block}
.ask{display:flex;gap:8px}.ask input{flex:1;font:inherit;padding:12px 14px;border:1px solid var(--line);border-radius:8px;background:var(--panel);color:var(--fg)}
button{font:inherit;border:1px solid var(--line);background:var(--panel);color:var(--fg);border-radius:8px;padding:9px 14px;cursor:pointer}
button.p{background:var(--acc);border-color:var(--acc);color:#fff}button:disabled{opacity:.5;cursor:default}
.ex{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}.ex button{font-size:12.5px;padding:4px 10px;border-radius:99px;color:var(--fg2)}
.res{margin-top:16px;display:grid;gap:14px}.status{display:flex;align-items:center;gap:12px;flex-wrap:wrap;padding:12px 14px;border-radius:8px}
.status.answer{background:var(--oks)}.status.unsure{background:var(--warns)}.status.fail{background:var(--bads)}
.status b{font-size:16px}.bar{flex:0 0 160px;height:8px;background:var(--line);border-radius:4px;overflow:hidden}.bar span{display:block;height:100%}
textarea{width:100%;min-height:84px;font:13px/1.5 ui-monospace,Menlo,monospace;background:var(--code);color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:10px}
.tw{overflow:auto;max-height:420px;border:1px solid var(--line);border-radius:8px}table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}th{position:sticky;top:0;background:var(--panel);font-weight:600;font-size:12px;color:var(--mut)}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}.muted{color:var(--mut);font-size:13px}.alt{font:12.5px ui-monospace,Menlo,monospace;background:var(--code);border-radius:6px;padding:8px;margin-top:6px;white-space:pre-wrap}
.lbl{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);margin-bottom:4px}.toast{color:var(--ok);font-size:13px}
.hist{font-size:13px}.hist div{padding:5px 0;border-top:1px solid var(--line);cursor:pointer}
.spin{display:inline-block;width:14px;height:14px;border:2px solid var(--line);border-top-color:var(--acc);border-radius:50%;animation:s 1s linear infinite;vertical-align:-2px}@keyframes s{to{transform:rotate(360deg)}}
</style></head><body>
<header><b>Datenabfrage</b><span class="pill"><i></i>Modell läuft lokal</span><span class="pill">nur lesender Zugriff</span>
<span class="sp"></span><span class="muted" id="who"></span></header>
<main>
<aside class="side">
 <div class="card"><h3>Datenbank</h3><div id="tables"></div></div>
 <div class="card"><h3>Glossar</h3><div id="gloss"></div></div>
 <div class="card"><h3>Verlauf</h3><div class="hist" id="hist"><span class="muted">Noch keine Fragen.</span></div></div>
</aside>
<section>
 <div class="card">
  <form class="ask" id="f"><input id="q" placeholder="Frage an die Daten, z.B. Wie viele offene Schäden gibt es?" autocomplete="off"><button class="p" id="go">Fragen</button></form>
  <div class="ex" id="ex"></div>
 </div>
 <div class="res" id="res"></div>
</section></main>
<script>
const $=s=>document.querySelector(s);const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
let META=null,CUR=null;const H=[];
const fmtv=v=>v===null?'<span class="muted">leer</span>':typeof v==='number'?(Number.isInteger(v)?v.toLocaleString('de-CH'):v.toLocaleString('de-CH',{maximumFractionDigits:2})):esc(v);
async function api(p,b){const r=await fetch(p,b?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)}:{});return r.json();}
function table(cols,rows,n){if(!rows)return'';return`<div class="tw"><table><tr>${cols.map(c=>`<th>${esc(c)}</th>`).join('')}</tr>${rows.map(r=>`<tr>${r.map(v=>`<td>${fmtv(v)}</td>`).join('')}</tr>`).join('')}</table></div>${n>rows.length?`<div class="muted">${rows.length} von ${n} Zeilen angezeigt</div>`:''}`;}
function render(a){CUR=a;const pct=Math.round(a.confidence*100);
 const cls=a.decision==='answer'?'answer':(a.rows&&!a.error?'unsure':'fail');
 const title=a.decision==='answer'?'Antwort':(a.rows&&!a.error?'Unsicher, bitte prüfen':'Keine verlässliche Antwort');
 const col=a.decision==='answer'?'var(--ok)':'var(--warn)';
 $('#res').innerHTML=`<div class="card"><div class="status ${cls}"><b>${title}</b><div class="bar"><span style="width:${pct}%;background:${col}"></span></div>
 <span>${a.votes} von ${a.n_voters} Antworten des Modells führen zum selben Resultat</span><span class="sp"></span><span class="muted">${a.latency_s.toFixed(1)} s${a.retried?' · nach Fehlerkorrektur':''}</span></div>
 ${a.decision!=='answer'?`<p class="muted">Das System ist sich nicht sicher genug (Schwelle ${Math.round(META.threshold*100)} %). Prüfen Sie das SQL oder wenden Sie sich an das BI-Team.${a.degenerate?' Die Abfrage liefert kein Resultat; oft ist ein Filterwert falsch.':''}</p>`:''}
 <div class="lbl" style="margin-top:12px">Resultat</div>${a.error?`<p style="color:var(--bad)">${esc(a.error)}</p>`:table(a.columns||[],a.rows,a.n_rows)}
 <div class="lbl" style="margin-top:14px">SQL (bearbeitbar)</div><textarea id="sql">${esc(a.sql)}</textarea>
 <div class="row" style="margin-top:8px"><button class="p" id="ok">✓ Bestätigen</button><button id="run">▶ Geändertes SQL ausführen</button><button id="no">✕ Falsch</button><span class="toast" id="toast"></span></div>
 ${a.alternatives.length?`<details style="margin-top:10px"><summary>Andere Antworten des Modells (${a.alternatives.length})</summary>${a.alternatives.map(x=>`<div class="alt">${x.votes}× ${esc(x.sql)}</div>`).join('')}</details>`:''}
 </div>`;
 $('#ok').onclick=()=>fb('confirmed');$('#no').onclick=()=>fb('rejected');
 $('#run').onclick=async()=>{const r=await api('/api/run',{sql:$('#sql').value});CUR={...CUR,...r,edited:true};
   const box=$('#res .card');const t=box.querySelector('.tw, p[style]');const html=r.error?`<p style="color:var(--bad)">${esc(r.error)}</p>`:table(r.columns||[],r.rows,r.n_rows);
   if(t)t.outerHTML=html;$('#toast').textContent='Ausgeführt. Mit «Bestätigen» als Korrektur speichern.';};
}
async function fb(verdict){const sql=$('#sql').value;const v=verdict==='confirmed'&&sql.trim()!==CUR.sql.trim()?'corrected':verdict;
 const r=await api('/api/feedback',{id:CUR.id,verdict:v,question:CUR.question,sql,model_sql:CUR.sql});
 $('#toast').textContent=r.ok?(v==='rejected'?'Als falsch markiert.':`Gespeichert als Trainingsbeispiel (${r.stored} insgesamt).`):r.error;}
async function go(q){if(!q)return;$('#q').value=q;$('#go').disabled=true;
 $('#res').innerHTML=`<div class="card"><span class="spin"></span> Das Modell schreibt mehrere Abfragen und vergleicht die Resultate …</div>`;
 const a=await api('/api/ask',{question:q});$('#go').disabled=false;
 if(a.error){$('#res').innerHTML=`<div class="card" style="color:var(--bad)">${esc(a.error)}</div>`;return;}
 render(a);H.unshift(a);$('#hist').innerHTML=H.slice(0,12).map((h,i)=>`<div data-i="${i}">${h.decision==='answer'?'✓':'?'} ${esc(h.question)}</div>`).join('');
 $('#hist').querySelectorAll('div').forEach(d=>d.onclick=()=>render(H[+d.dataset.i]));}
$('#f').onsubmit=e=>{e.preventDefault();go($('#q').value.trim());};
api('/api/meta').then(m=>{META=m;$('#who').textContent=`${m.user} · ${m.model}`;
 $('#tables').innerHTML=m.tables.map(t=>`<details><summary>${t.name} <span class="muted">(${t.rows})</span></summary><div class="d">${esc(t.desc)}</div>${t.columns.map(c=>`<div class="col"><code>${c.name}</code> <span>${esc(c.desc)}</span></div>`).join('')}</details>`).join('');
 $('#gloss').innerHTML=Object.entries(m.glossary).map(([k,v])=>`<div class="gl"><b>${esc(k)}</b>${esc(v)}</div>`).join('');
 $('#ex').innerHTML=m.examples.map(e=>`<button type="button">${esc(e)}</button>`).join('');$('#ex').querySelectorAll('button').forEach(b=>b.onclick=()=>go(b.textContent));});
</script></body></html>"""

if __name__ == "__main__":
    main()
