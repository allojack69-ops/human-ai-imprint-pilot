from __future__ import annotations
import csv, io, json, math, os, random, time, uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from psycopg import connect
from psycopg.rows import dict_row

BASE = Path(__file__).resolve().parent
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is required")

QUESTIONS = json.loads((BASE/"data/questions.json").read_text(encoding="utf-8"))
AI_QUESTIONS = json.loads((BASE/"data/ai_questions.json").read_text(encoding="utf-8"))
QMAP = {q["id"]: q for q in QUESTIONS}
AIMAP = {q["id"]: q for q in AI_QUESTIONS}
ADMIN_KEY = os.environ.get("ADMIN_KEY", "change-me")

app = FastAPI(title="Human–AI Imprint Pilot v1.3")
app.mount("/static", StaticFiles(directory=BASE/"static"), name="static")

def db():
    return connect(DATABASE_URL, row_factory=dict_row, autocommit=False)

def init_db():
    con = db()
    cur = con.cursor()
    statements = [
        """CREATE TABLE IF NOT EXISTS participants(
          id TEXT PRIMARY KEY, alias TEXT, age_confirmed INTEGER NOT NULL,
          consent INTEGER NOT NULL, created_at DOUBLE PRECISION NOT NULL,
          finished_human_at DOUBLE PRECISION, finished_ai_at DOUBLE PRECISION)""",
        """CREATE TABLE IF NOT EXISTS answers(
          id BIGSERIAL PRIMARY KEY, participant_id TEXT NOT NULL, question_id TEXT NOT NULL,
          answer_id TEXT, answer_text TEXT, reaction_ms INTEGER, changed_answer INTEGER DEFAULT 0,
          order_index INTEGER, created_at DOUBLE PRECISION NOT NULL,
          UNIQUE(participant_id, question_id))""",
        """CREATE TABLE IF NOT EXISTS ai_answers(
          id BIGSERIAL PRIMARY KEY, participant_id TEXT NOT NULL, probe_id TEXT NOT NULL,
          model_name TEXT, answer_text TEXT NOT NULL DEFAULT '', choice_id TEXT, reason_text TEXT,
          created_at DOUBLE PRECISION NOT NULL, UNIQUE(participant_id, probe_id))""",
        """CREATE TABLE IF NOT EXISTS fingerprints(
          participant_id TEXT PRIMARY KEY, vector_json TEXT NOT NULL, created_at DOUBLE PRECISION NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS ai_fingerprints(
          participant_id TEXT PRIMARY KEY, model_name TEXT NOT NULL, vector_json TEXT NOT NULL,
          created_at DOUBLE PRECISION NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS ai_metadata(
          participant_id TEXT PRIMARY KEY, model_name TEXT NOT NULL,
          memory_status TEXT NOT NULL DEFAULT 'unknown',
          custom_instructions TEXT NOT NULL DEFAULT 'unknown',
          usage_duration TEXT NOT NULL DEFAULT 'unknown',
          fresh_chat INTEGER NOT NULL DEFAULT 1, created_at DOUBLE PRECISION NOT NULL)"""
    ]
    for sql in statements:
        cur.execute(sql)
    con.commit()
    con.close()

init_db()

class StartPayload(BaseModel):
    alias: Optional[str] = ""
    age_confirmed: bool
    consent: bool

class AnswerPayload(BaseModel):
    participant_id: str
    question_id: str
    answer_id: Optional[str] = None
    answer_text: Optional[str] = ""
    reaction_ms: int = 0
    changed_answer: bool = False

class AIChoice(BaseModel):
    choice: str
    reason: Optional[str] = ""

class AIPayload(BaseModel):
    participant_id: str
    model_name: str
    memory_status: str = "unknown"
    custom_instructions: str = "unknown"
    usage_duration: str = "unknown"
    fresh_chat: bool = True
    answers: dict[str, AIChoice]

def participant_or_404(pid: str):
    con = db()
    row = con.execute("SELECT * FROM participants WHERE id=%s", (pid,)).fetchone()
    con.close()
    if not row:
        raise HTTPException(404, "participant not found")
    return row

def answered_ids(pid: str):
    con = db()
    rows = con.execute("SELECT question_id FROM answers WHERE participant_id=%s ORDER BY order_index", (pid,)).fetchall()
    con.close()
    return [r["question_id"] for r in rows]

def trait_counts(pid: str):
    counts = {}
    for qid in answered_ids(pid):
        q = QMAP.get(qid)
        if q:
            counts[q["trait"]] = counts.get(q["trait"], 0) + 1
    return counts

def choose_next(pid: str):
    done = answered_ids(pid)
    done_set = set(done)
    easy = [q for q in QUESTIONS if q["level"]=="easy" and q["id"] not in done_set]
    medium = [q for q in QUESTIONS if q["level"]=="medium" and q["id"] not in done_set]
    hard = [q for q in QUESTIONS if q["level"]=="hard" and q["id"] not in done_set]
    n = len(done)

    # 18 easy + 4 medium + 2 hard.
    if n < 18 and easy:
        counts = trait_counts(pid)
        min_count = min([counts.get(q["trait"],0) for q in easy] or [0])
        candidates = [q for q in easy if counts.get(q["trait"],0) == min_count]
        return random.choice(candidates)
    if n < 22 and medium:
        return random.choice(medium)
    if n < 24 and hard:
        return hard[0]
    return None

def aggregate_vector(rows, mapping, id_field, choice_field):
    sums, counts = {}, {}
    for r in rows:
        q = mapping.get(r[id_field])
        if not q or q.get("free_text"):
            continue
        opt = next((o for o in q.get("options",[]) if o["id"] == r[choice_field]), None)
        if not opt:
            continue
        for k,v in opt.get("v",{}).items():
            sums[k] = sums.get(k,0.0) + float(v)
            counts[k] = counts.get(k,0) + 1
    return {k: round(sums[k]/counts[k], 4) for k in sums if counts[k]}

def compute_human_fingerprint(pid: str):
    con = db()
    rows = con.execute("SELECT question_id,answer_id FROM answers WHERE participant_id=%s", (pid,)).fetchall()
    con.close()
    return aggregate_vector(rows, QMAP, "question_id", "answer_id")

def compute_ai_fingerprint(pid: str):
    con = db()
    rows = con.execute("SELECT probe_id,choice_id FROM ai_answers WHERE participant_id=%s", (pid,)).fetchall()
    con.close()
    return aggregate_vector(rows, AIMAP, "probe_id", "choice_id")

def cosine(a, b):
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    x = [float(a[k]) for k in keys]
    y = [float(b[k]) for k in keys]
    nx = math.sqrt(sum(v*v for v in x))
    ny = math.sqrt(sum(v*v for v in y))
    if nx == 0 or ny == 0:
        return None
    return sum(i*j for i,j in zip(x,y))/(nx*ny)

def sign_agreement(a,b):
    keys = sorted(set(a)&set(b))
    if not keys:
        return None
    def s(v):
        return 1 if v > 0.15 else (-1 if v < -0.15 else 0)
    return sum(s(a[k])==s(b[k]) for k in keys)/len(keys)

def l1_distance(a,b):
    keys = sorted(set(a)&set(b))
    if not keys:
        return None
    return sum(abs(a[k]-b[k]) for k in keys)/len(keys)

@app.get("/", response_class=HTMLResponse)
def home():
    return FileResponse(BASE/"static/index.html")

@app.post("/api/start")
def start(p: StartPayload):
    if not p.age_confirmed or not p.consent:
        raise HTTPException(400, "Consent and adult confirmation required")
    pid = uuid.uuid4().hex[:16]
    con = db()
    con.execute("INSERT INTO participants(id,alias,age_confirmed,consent,created_at) VALUES(%s,%s,%s,%s,%s)",
                (pid, p.alias or "", 1,1,time.time()))
    con.commit(); con.close()
    return {"participant_id":pid}

@app.get("/api/next/{pid}")
def next_question(pid: str):
    participant_or_404(pid)
    q = choose_next(pid)
    if not q:
        vec = compute_human_fingerprint(pid)
        con = db()
        con.execute("UPDATE participants SET finished_human_at=COALESCE(finished_human_at,%s) WHERE id=%s", (time.time(),pid))
        con.execute("""INSERT INTO fingerprints(participant_id,vector_json,created_at) VALUES(%s,%s,%s)
                    ON CONFLICT(participant_id) DO UPDATE SET
                    vector_json=EXCLUDED.vector_json, created_at=EXCLUDED.created_at""",
                    (pid,json.dumps(vec,ensure_ascii=False),time.time()))
        con.commit(); con.close()
        return {"done":True}
    public = {k:q[k] for k in q if k not in ("v","trait")}
    if "options" in public:
        public["options"] = [{"id":o["id"],"text":o["text"]} for o in public["options"]]
    return {"done":False,"question":public,"progress":{"answered":len(answered_ids(pid)),"target":24}}

@app.post("/api/answer")
def answer(p: AnswerPayload):
    participant_or_404(p.participant_id)
    q = QMAP.get(p.question_id)
    if not q:
        raise HTTPException(400, "unknown question")
    done = answered_ids(p.participant_id)
    if p.question_id in done:
        raise HTTPException(409, "already answered")
    if not q.get("free_text"):
        allowed = {o["id"] for o in q["options"]}
        if p.answer_id not in allowed:
            raise HTTPException(400, "invalid option")
    con = db()
    con.execute("""INSERT INTO answers(participant_id,question_id,answer_id,answer_text,reaction_ms,changed_answer,order_index,created_at)
                   VALUES(%s,%s,%s,%s,%s,%s,%s,%s)""",
                (p.participant_id,p.question_id,p.answer_id,p.answer_text or "",max(0,p.reaction_ms),
                 int(p.changed_answer),len(done)+1,time.time()))
    con.commit(); con.close()
    return {"ok":True}

@app.get("/api/ai-pack/{pid}")
def ai_pack(pid: str):
    participant_or_404(pid)
    lines = []
    for q in AI_QUESTIONS:
        opts = " | ".join(f'{o["id"]}: {o["text"]}' for o in q["options"])
        lines.append(f'{q["id"]}. {q["text"]}\nOPTIONS: {opts}')
    prompt = """You are participating in a behavioral experiment.
Answer each scenario independently and naturally, based on how you would reason in your current personalized state.

Important:
- This prompt should be run in a NEW normal chat in the user's usual personal account.
- Do not use Temporary/Incognito mode if that would disable normal personalization or memory.
- Do not ask for or use the participant's answers from the human test.
- Do not try to imitate the participant.
- Do not mention or reveal private facts about the user.
- Do not optimize for appearing unique.
- Choose exactly one listed option for every item.
- Give a short reason (1-3 sentences).
- Return ONLY valid JSON in exactly this shape:
{
  "answers": {
    "P01": {"choice":"A","reason":"..."},
    "P02": {"choice":"B","reason":"..."}
  }
}

SCENARIOS:
""" + "\n\n".join(lines)
    public = []
    for q in AI_QUESTIONS:
        public.append({
            "id":q["id"], "text":q["text"],
            "options":[{"id":o["id"],"text":o["text"]} for o in q["options"]]
        })
    return {"participant_id":pid,"questions":public,"copy_prompt":prompt}

@app.post("/api/ai-submit")
def ai_submit(p: AIPayload):
    participant_or_404(p.participant_id)
    missing = [q["id"] for q in AI_QUESTIONS if q["id"] not in p.answers]
    if missing:
        raise HTTPException(400, f"Missing probes: {', '.join(missing)}")

    normalized = {}
    for q in AI_QUESTIONS:
        aid = q["id"]
        item = p.answers[aid]
        allowed = {o["id"] for o in q["options"]}
        choice = item.choice.strip().upper()
        if choice not in allowed:
            raise HTTPException(400, f"Invalid choice for {aid}: {choice}")
        normalized[aid] = (choice, (item.reason or "").strip())

    con = db()
    for aid,(choice,reason) in normalized.items():
        con.execute("""INSERT INTO ai_answers
                       (participant_id,probe_id,model_name,answer_text,choice_id,reason_text,created_at)
                       VALUES(%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT(participant_id,probe_id) DO UPDATE SET
                         model_name=EXCLUDED.model_name,
                         answer_text=EXCLUDED.answer_text,
                         choice_id=EXCLUDED.choice_id,
                         reason_text=EXCLUDED.reason_text,
                         created_at=EXCLUDED.created_at""",
                    (p.participant_id,aid,p.model_name,reason,choice,reason,time.time()))
    con.execute("UPDATE participants SET finished_ai_at=%s WHERE id=%s", (time.time(),p.participant_id))
    con.execute("""INSERT INTO ai_metadata
                   (participant_id,model_name,memory_status,custom_instructions,usage_duration,fresh_chat,created_at)
                   VALUES(%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT(participant_id) DO UPDATE SET
                     model_name=EXCLUDED.model_name,
                     memory_status=EXCLUDED.memory_status,
                     custom_instructions=EXCLUDED.custom_instructions,
                     usage_duration=EXCLUDED.usage_duration,
                     fresh_chat=EXCLUDED.fresh_chat,
                     created_at=EXCLUDED.created_at""",
                (p.participant_id,p.model_name,p.memory_status,p.custom_instructions,
                 p.usage_duration,int(p.fresh_chat),time.time()))
    con.commit(); con.close()

    vec = compute_ai_fingerprint(p.participant_id)
    con = db()
    con.execute("""INSERT INTO ai_fingerprints(participant_id,model_name,vector_json,created_at)
                   VALUES(%s,%s,%s,%s)
                   ON CONFLICT(participant_id) DO UPDATE SET
                     model_name=EXCLUDED.model_name,
                     vector_json=EXCLUDED.vector_json,
                     created_at=EXCLUDED.created_at""",
                (p.participant_id,p.model_name,json.dumps(vec,ensure_ascii=False),time.time()))
    human_row = con.execute("SELECT vector_json FROM fingerprints WHERE participant_id=%s", (p.participant_id,)).fetchone()
    con.commit(); con.close()

    metrics = {}
    if human_row:
        hv = json.loads(human_row["vector_json"])
        metrics = {
            "cosine_similarity": cosine(hv,vec),
            "sign_agreement": sign_agreement(hv,vec),
            "mean_absolute_distance": l1_distance(hv,vec),
            "shared_dimensions": len(set(hv)&set(vec))
        }
    return {"ok":True,"pair_metrics":metrics}

def admin_check(key: Optional[str], header_key: Optional[str]):
    supplied = header_key or key
    if supplied != ADMIN_KEY:
        raise HTTPException(403, "bad admin key")

def model_family(name: str):
    n=(name or "").strip().lower()
    if "chatgpt" in n or "gpt" in n or "openai" in n: return "openai"
    if "claude" in n or "anthropic" in n: return "anthropic"
    if "gemini" in n or "google" in n: return "google"
    if "grok" in n or "xai" in n: return "xai"
    if "deepseek" in n: return "deepseek"
    if "llama" in n or "meta" in n: return "meta"
    return n or "unknown"

def all_vectors():
    con = db()
    hr = con.execute("SELECT participant_id,vector_json FROM fingerprints").fetchall()
    ar = con.execute("""SELECT f.participant_id,f.model_name,f.vector_json,
                               m.memory_status,m.custom_instructions,m.usage_duration,m.fresh_chat
                        FROM ai_fingerprints f LEFT JOIN ai_metadata m ON f.participant_id=m.participant_id""").fetchall()
    con.close()
    humans = {r["participant_id"]:json.loads(r["vector_json"]) for r in hr}
    ais = {r["participant_id"]:{
        "model_name":r["model_name"],
        "model_family":model_family(r["model_name"]),
        "vector":json.loads(r["vector_json"]),
        "memory_status":r["memory_status"] or "unknown",
        "custom_instructions":r["custom_instructions"] or "unknown",
        "usage_duration":r["usage_duration"] or "unknown",
        "fresh_chat":bool(r["fresh_chat"]) if r["fresh_chat"] is not None else None
    } for r in ar}
    return humans,ais

def study_summary():
    con = db()
    total = con.execute("SELECT COUNT(*) n FROM participants").fetchone()["n"]
    human_done = con.execute("SELECT COUNT(*) n FROM participants WHERE finished_human_at IS NOT NULL").fetchone()["n"]
    ai_done = con.execute("SELECT COUNT(*) n FROM participants WHERE finished_ai_at IS NOT NULL").fetchone()["n"]
    rt = con.execute("SELECT AVG(reaction_ms) x FROM answers WHERE reaction_ms>0").fetchone()["x"]
    con.close()

    humans,ais = all_vectors()
    paired = sorted(set(humans)&set(ais))
    matched, mismatched, within_family_mismatch = [], [], []
    rank_hits = rank_total = 0
    within_rank_hits = within_rank_total = 0
    pair_rows = []

    for pid in paired:
        s = cosine(humans[pid],ais[pid]["vector"])
        if s is not None: matched.append(s)

        candidates=[]
        family_candidates=[]
        fam=ais[pid]["model_family"]
        for aid in paired:
            cs=cosine(humans[pid],ais[aid]["vector"])
            if cs is None: continue
            candidates.append((aid,cs))
            if aid != pid:
                mismatched.append(cs)
                if ais[aid]["model_family"] == fam:
                    within_family_mismatch.append(cs)
            if ais[aid]["model_family"] == fam:
                family_candidates.append((aid,cs))

        candidates.sort(key=lambda x:x[1],reverse=True)
        if candidates:
            rank_total += 1
            rank_hits += int(candidates[0][0] == pid)

        family_candidates.sort(key=lambda x:x[1],reverse=True)
        # Meaningful only if there is at least one competitor from same family.
        if len(family_candidates) >= 2:
            within_rank_total += 1
            within_rank_hits += int(family_candidates[0][0] == pid)

        pair_rows.append({
            "participant_id":pid,
            "model_name":ais[pid]["model_name"],
            "model_family":fam,
            "memory_status":ais[pid]["memory_status"],
            "custom_instructions":ais[pid]["custom_instructions"],
            "usage_duration":ais[pid]["usage_duration"],
            "fresh_chat":ais[pid]["fresh_chat"],
            "matched_cosine":s,
            "sign_agreement":sign_agreement(humans[pid],ais[pid]["vector"]),
            "mean_absolute_distance":l1_distance(humans[pid],ais[pid]["vector"]),
            "top1_identified":bool(candidates and candidates[0][0]==pid),
            "within_family_top1":(bool(family_candidates and family_candidates[0][0]==pid)
                                  if len(family_candidates)>=2 else None)
        })

    matched_mean=sum(matched)/len(matched) if matched else None
    mismatch_mean=sum(mismatched)/len(mismatched) if mismatched else None
    within_mean=sum(within_family_mismatch)/len(within_family_mismatch) if within_family_mismatch else None

    return {
        "participants_total":total,
        "human_completed":human_done,
        "ai_completed":ai_done,
        "paired_n":len(paired),
        "mean_reaction_ms":rt,
        "matched_similarity_mean":matched_mean,
        "mismatched_similarity_mean":mismatch_mean,
        "similarity_gap":(matched_mean-mismatch_mean if matched_mean is not None and mismatch_mean is not None else None),
        "within_family_mismatched_mean":within_mean,
        "within_family_gap":(matched_mean-within_mean if matched_mean is not None and within_mean is not None else None),
        "top1_identification_accuracy":(rank_hits/rank_total if rank_total else None),
        "within_family_top1_accuracy":(within_rank_hits/within_rank_total if within_rank_total else None),
        "within_family_rank_n":within_rank_total,
        "pairs":pair_rows
    }

@app.get("/admin/summary")
def admin_summary(key: Optional[str]=None, x_admin_key: Optional[str]=Header(default=None)):
    admin_check(key,x_admin_key)
    return study_summary()

@app.get("/admin", response_class=HTMLResponse)
def admin_page(key: Optional[str]=None, x_admin_key: Optional[str]=Header(default=None)):
    admin_check(key,x_admin_key)
    return FileResponse(BASE/"static/admin.html")

@app.get("/admin/export.csv")
def export_csv(key: Optional[str]=None, x_admin_key: Optional[str]=Header(default=None)):
    admin_check(key,x_admin_key)
    con = db()
    rows = con.execute("""
      SELECT p.id participant_id,p.alias,p.created_at,p.finished_human_at,p.finished_ai_at,
             a.question_id,a.answer_id,a.answer_text,a.reaction_ms,a.changed_answer,a.order_index
      FROM participants p LEFT JOIN answers a ON p.id=a.participant_id
      ORDER BY p.created_at,a.order_index
    """).fetchall()
    con.close()
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(list(rows[0].keys()) if rows else ["participant_id"])
    for r in rows: w.writerow([r[k] for k in r.keys()])
    return StreamingResponse(iter([out.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition":"attachment; filename=human_answers.csv"})

@app.get("/admin/export_ai.csv")
def export_ai_csv(key: Optional[str]=None, x_admin_key: Optional[str]=Header(default=None)):
    admin_check(key,x_admin_key)
    con = db()
    rows = con.execute("""
      SELECT participant_id,probe_id,model_name,choice_id,reason_text,created_at
      FROM ai_answers ORDER BY participant_id,probe_id
    """).fetchall()
    con.close()
    out = io.StringIO(); w=csv.writer(out)
    w.writerow(list(rows[0].keys()) if rows else ["participant_id"])
    for r in rows: w.writerow([r[k] for k in r.keys()])
    return StreamingResponse(iter([out.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition":"attachment; filename=ai_answers.csv"})

@app.get("/admin/export_fingerprints.json")
def export_fp(key: Optional[str]=None, x_admin_key: Optional[str]=Header(default=None)):
    admin_check(key,x_admin_key)
    humans,ais = all_vectors()
    data=[]
    for pid in sorted(set(humans)|set(ais)):
        data.append({
            "participant_id":pid,
            "human":humans.get(pid),
            "ai":(ais.get(pid) or {}).get("vector"),
            "model_name":(ais.get(pid) or {}).get("model_name"),
            "model_family":(ais.get(pid) or {}).get("model_family"),
            "memory_status":(ais.get(pid) or {}).get("memory_status"),
            "custom_instructions":(ais.get(pid) or {}).get("custom_instructions"),
            "usage_duration":(ais.get(pid) or {}).get("usage_duration"),
            "fresh_chat":(ais.get(pid) or {}).get("fresh_chat")
        })
    return JSONResponse(data)

@app.get("/health")
def health():
    return {"ok":True,"version":"1.3"}
