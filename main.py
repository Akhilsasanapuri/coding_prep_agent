"""Coding Prep Agent - lazy MVP (single-file FastAPI + SQLite).

Covers the doc P0 loop: onboarding -> adaptive plan -> DSA tracker -> agent chat
-> readiness, plus a simple mock loop. Rule-based plan/agent; LLM used only if
OPENAI_API_KEY is set (deterministic fallback otherwise).

ponytail: SQLite not Postgres, in-proc not Redis, rule-based not LangGraph, no
RAG/pgvector, no resume tailoring. Add those when you upload docs to retrieve,
need JD tailoring, or hit multi-user scale. Data access + answer() are the two
seams to swap later.
"""
import os, sqlite3, hashlib, hmac, json, base64, time, datetime as dt
from contextlib import contextmanager
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

HERE = os.path.dirname(__file__)

# ponytail: 4-line stdlib .env loader instead of the python-dotenv dep
_envf = os.path.join(HERE, ".env")
if os.path.exists(_envf):
    for _l in open(_envf, encoding="utf-8"):
        _l = _l.strip()
        if _l and not _l.startswith("#") and "=" in _l:
            _k, _v = _l.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

DB = os.path.join(HERE, "placement.db")
SECRET = os.environ.get("APP_SECRET", "dev-secret-change-me").encode()

app = FastAPI(title="Coding Prep Agent")

# ---------------------------------------------------------------- db
@contextmanager
def db():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()

def init_db():
    with db() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY, email TEXT UNIQUE, pw TEXT, name TEXT, created_at TEXT);
        CREATE TABLE IF NOT EXISTS profiles(
            user_id INTEGER PRIMARY KEY, target_role TEXT, target_companies TEXT,
            daily_minutes INTEGER, interview_date TEXT, skill_level TEXT,
            strengths TEXT, weak_areas TEXT);
        CREATE TABLE IF NOT EXISTS tasks(
            id INTEGER PRIMARY KEY, user_id INTEGER, title TEXT, category TEXT,
            priority TEXT, due_date TEXT, status TEXT DEFAULT 'pending',
            source_agent TEXT, created_at TEXT);
        CREATE TABLE IF NOT EXISTS problems(
            id INTEGER PRIMARY KEY, user_id INTEGER, title TEXT, platform TEXT,
            url TEXT, difficulty TEXT, tags TEXT, status TEXT, attempts INTEGER DEFAULT 1,
            notes TEXT, last_revised_at TEXT, created_at TEXT);
        CREATE TABLE IF NOT EXISTS mock_sessions(
            id INTEGER PRIMARY KEY, user_id INTEGER, type TEXT, difficulty TEXT,
            score REAL, feedback_summary TEXT, created_at TEXT);
        CREATE TABLE IF NOT EXISTS mock_turns(
            id INTEGER PRIMARY KEY, session_id INTEGER, question TEXT, answer TEXT,
            feedback TEXT, score REAL);
        CREATE INDEX IF NOT EXISTS ix_tasks ON tasks(user_id, due_date, status);
        CREATE INDEX IF NOT EXISTS ix_problems ON problems(user_id, status);
        """)

# ---------------------------------------------------------------- auth
def hash_pw(pw: str) -> str:
    salt = os.urandom(16)
    h = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, 100_000)
    return salt.hex() + ":" + h.hex()

def verify_pw(pw: str, stored: str) -> bool:
    salt_hex, h_hex = stored.split(":")
    h = hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt_hex), 100_000)
    return hmac.compare_digest(h.hex(), h_hex)

def make_token(user_id: int) -> str:
    body = base64.urlsafe_b64encode(json.dumps({"uid": user_id, "t": int(time.time())}).encode())
    sig = hmac.new(SECRET, body, hashlib.sha256).digest()
    return body.decode() + "." + base64.urlsafe_b64encode(sig).decode()

def read_token(token: str) -> int:
    try:
        body_b64, sig_b64 = token.split(".")
        body = body_b64.encode()
        expect = hmac.new(SECRET, body, hashlib.sha256).digest()
        if not hmac.compare_digest(base64.urlsafe_b64decode(sig_b64), expect):
            raise ValueError
        return json.loads(base64.urlsafe_b64decode(body))["uid"]
    except Exception:
        raise HTTPException(401, "invalid token")

def current_user(authorization: str = Header(None)) -> int:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing token")
    return read_token(authorization[7:])

# ---------------------------------------------------------------- models
class Signup(BaseModel):
    email: str; password: str; name: str = ""
class Login(BaseModel):
    email: str; password: str
class Profile(BaseModel):
    target_role: str = ""; target_companies: str = ""; daily_minutes: int = 60
    interview_date: Optional[str] = None; skill_level: str = "intermediate"
    strengths: str = ""; weak_areas: str = ""
class ProblemIn(BaseModel):
    title: str; platform: str = "LeetCode"; url: str = ""; difficulty: str = "Medium"
    tags: str = ""; status: str = "solved"; notes: str = ""
class TaskPatch(BaseModel):
    status: str
class ChatIn(BaseModel):
    message: str; mode: str = "planning"
class MockStart(BaseModel):
    type: str = "DSA"; difficulty: str = "Medium"
class MockAnswer(BaseModel):
    answer: str

# ---------------------------------------------------------------- auth routes
@app.post("/api/auth/signup")
def signup(s: Signup):
    with db() as con:
        try:
            cur = con.execute("INSERT INTO users(email,pw,name,created_at) VALUES(?,?,?,?)",
                              (s.email.lower(), hash_pw(s.password), s.name, dt.datetime.utcnow().isoformat()))
        except sqlite3.IntegrityError:
            raise HTTPException(409, "email already registered")
        return {"token": make_token(cur.lastrowid)}

@app.post("/api/auth/login")
def login(l: Login):
    with db() as con:
        u = con.execute("SELECT id,pw FROM users WHERE email=?", (l.email.lower(),)).fetchone()
    if not u or not verify_pw(l.password, u["pw"]):
        raise HTTPException(401, "invalid credentials")
    return {"token": make_token(u["id"])}

# ---------------------------------------------------------------- profile
@app.get("/api/profile")
def get_profile(uid: int = Depends(current_user)):
    with db() as con:
        p = con.execute("SELECT * FROM profiles WHERE user_id=?", (uid,)).fetchone()
    return dict(p) if p else None

@app.put("/api/profile")
def put_profile(p: Profile, uid: int = Depends(current_user)):
    with db() as con:
        con.execute("""INSERT INTO profiles(user_id,target_role,target_companies,daily_minutes,
            interview_date,skill_level,strengths,weak_areas) VALUES(?,?,?,?,?,?,?,?)
            ON CONFLICT(user_id) DO UPDATE SET target_role=excluded.target_role,
            target_companies=excluded.target_companies, daily_minutes=excluded.daily_minutes,
            interview_date=excluded.interview_date, skill_level=excluded.skill_level,
            strengths=excluded.strengths, weak_areas=excluded.weak_areas""",
            (uid, p.target_role, p.target_companies, p.daily_minutes, p.interview_date,
             p.skill_level, p.strengths, p.weak_areas))
    return {"ok": True}

# ---------------------------------------------------------------- problems
@app.post("/api/problems")
def add_problem(pr: ProblemIn, uid: int = Depends(current_user)):
    now = dt.datetime.utcnow().isoformat()
    with db() as con:
        cur = con.execute("""INSERT INTO problems(user_id,title,platform,url,difficulty,tags,
            status,notes,last_revised_at,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (uid, pr.title, pr.platform, pr.url, pr.difficulty, pr.tags.lower(), pr.status,
             pr.notes, now, now))
    return {"id": cur.lastrowid}

@app.get("/api/problems")
def list_problems(uid: int = Depends(current_user)):
    with db() as con:
        rows = con.execute("SELECT * FROM problems WHERE user_id=? ORDER BY created_at DESC", (uid,)).fetchall()
    return [dict(r) for r in rows]

# ---------------------------------------------------------------- planner (rule-based)
DSA_TOPICS = ["arrays", "strings", "hashing", "two pointers", "binary search",
              "linked list", "stacks", "recursion", "trees", "graphs", "dp", "greedy"]

def weak_topics(uid: int) -> list[str]:
    """Weak = declared weak_areas, plus DSA topics with no solved problems."""
    with db() as con:
        prof = con.execute("SELECT weak_areas FROM profiles WHERE user_id=?", (uid,)).fetchone()
        solved_tags = con.execute(
            "SELECT tags FROM problems WHERE user_id=? AND status='solved'", (uid,)).fetchall()
    covered = set()
    for r in solved_tags:
        covered |= {t.strip() for t in (r["tags"] or "").split(",") if t.strip()}
    declared = [w.strip().lower() for w in (prof["weak_areas"] if prof else "").split(",") if w.strip()]
    uncovered = [t for t in DSA_TOPICS if t not in covered]
    # declared first (user knows best), then uncovered topics, de-duped, order-preserving
    return list(dict.fromkeys(declared + uncovered))

def generate_plan(uid: int) -> list[dict]:
    with db() as con:
        prof = con.execute("SELECT * FROM profiles WHERE user_id=?", (uid,)).fetchone()
    if not prof:
        raise HTTPException(400, "create a profile first")
    weak = weak_topics(uid)
    # days until interview (default 14), backwards plan from today
    days = 14
    if prof["interview_date"]:
        try:
            d = (dt.date.fromisoformat(prof["interview_date"]) - dt.date.today()).days
            days = max(1, min(d, 30))
        except ValueError:
            pass
    tasks = []
    today = dt.date.today()
    for i in range(days):
        due = (today + dt.timedelta(days=i)).isoformat()
        topic = weak[i % len(weak)] if weak else "arrays"
        tasks.append(("DSA", f"Solve 2-3 {topic} problems", "high" if i < days // 2 else "medium", due))
        if i % 2 == 0:
            tasks.append(("CS", f"Revise CS topic: {['OS','DBMS','CN','OOPS'][i//2 % 4]}", "medium", due))
        if i % 3 == 0:
            tasks.append(("Behavioral", "Prepare 1 STAR story", "low", due))
        if i % 4 == 3:
            tasks.append(("Mock", f"Take a {['DSA','behavioral','project'][i % 3]} mock", "high", due))
    now = dt.datetime.utcnow().isoformat()
    ids = []
    with db() as con:
        for cat, title, prio, due in tasks:
            cur = con.execute("""INSERT INTO tasks(user_id,title,category,priority,due_date,
                source_agent,created_at) VALUES(?,?,?,?,?,?,?)""",
                (uid, title, cat, prio, due, "planner", now))
            ids.append(cur.lastrowid)
    return ids

@app.post("/api/plans/generate")
def plans_generate(uid: int = Depends(current_user)):
    ids = generate_plan(uid)
    return {"createdTaskIds": ids, "count": len(ids)}

@app.get("/api/tasks")
def list_tasks(uid: int = Depends(current_user)):
    with db() as con:
        rows = con.execute(
            "SELECT * FROM tasks WHERE user_id=? ORDER BY due_date, priority", (uid,)).fetchall()
    return [dict(r) for r in rows]

@app.patch("/api/tasks/{task_id}")
def patch_task(task_id: int, patch: TaskPatch, uid: int = Depends(current_user)):
    with db() as con:
        cur = con.execute("UPDATE tasks SET status=? WHERE id=? AND user_id=?",
                          (patch.status, task_id, uid))
        if cur.rowcount == 0:
            raise HTTPException(404, "task not found")
    return {"ok": True}

# ---------------------------------------------------------------- readiness
def readiness(uid: int) -> dict:
    with db() as con:
        solved = con.execute("SELECT COUNT(*) c FROM problems WHERE user_id=? AND status='solved'", (uid,)).fetchone()["c"]
        tasks_done = con.execute("SELECT COUNT(*) c FROM tasks WHERE user_id=? AND status='done'", (uid,)).fetchone()["c"]
        tasks_total = con.execute("SELECT COUNT(*) c FROM tasks WHERE user_id=?", (uid,)).fetchone()["c"]
        mock_avg = con.execute("SELECT AVG(score) a FROM mock_sessions WHERE user_id=?", (uid,)).fetchone()["a"]
    # weighted score, each component capped at its weight
    dsa = min(solved / 75, 1.0) * 40            # 75 problems -> full 40 pts
    plan = (tasks_done / tasks_total if tasks_total else 0) * 30
    mock = ((mock_avg or 0) / 10) * 30          # mock scores are 0-10
    score = round(dsa + plan + mock)
    return {"score": score, "solved": solved, "tasks_done": tasks_done,
            "tasks_total": tasks_total, "mock_avg": round(mock_avg, 1) if mock_avg else None,
            "weak_topics": weak_topics(uid)[:6]}

@app.get("/api/readiness")
def get_readiness(uid: int = Depends(current_user)):
    return readiness(uid)

# ---------------------------------------------------------------- mock loop
MOCK_Q = {
    "DSA": "Given an array of integers, return indices of the two numbers that add to a target. Explain approach, complexity, and edge cases.",
    "behavioral": "Tell me about a time you handled a conflict in a team project. Use the STAR format.",
    "project": "Walk me through the hardest technical problem in your favorite project and how you solved it.",
    "system design basics": "Design a URL shortener. Cover the API, storage, and how you handle scale.",
}

@app.post("/api/mocks/start")
def mock_start(m: MockStart, uid: int = Depends(current_user)):
    q = MOCK_Q.get(m.type, MOCK_Q["DSA"])
    now = dt.datetime.utcnow().isoformat()
    with db() as con:
        cur = con.execute("INSERT INTO mock_sessions(user_id,type,difficulty,created_at) VALUES(?,?,?,?)",
                          (uid, m.type, m.difficulty, now))
        sid = cur.lastrowid
        con.execute("INSERT INTO mock_turns(session_id,question) VALUES(?,?)", (sid, q))
    return {"session_id": sid, "question": q}

def score_answer(answer: str) -> tuple[float, str]:
    """Cheap heuristic rubric: length + presence of structure/complexity keywords.
    ponytail: keyword heuristic; swap for LLM in answer()/here when a key is set."""
    a = answer.lower()
    pts, notes = 0.0, []
    if len(answer.split()) >= 40: pts += 3; notes.append("good depth")
    else: notes.append("answer is short: add more detail")
    if any(k in a for k in ["o(", "complexity", "time ", "space"]): pts += 3; notes.append("addressed complexity")
    else: notes.append("mention time/space complexity")
    if any(k in a for k in ["edge", "empty", "null", "corner"]): pts += 2; notes.append("considered edge cases")
    else: notes.append("call out edge cases")
    if any(k in a for k in ["situation", "task", "action", "result", "first", "then", "finally"]):
        pts += 2; notes.append("structured")
    else: notes.append("use a clear structure (e.g. STAR)")
    return min(pts, 10.0), "; ".join(notes)

@app.post("/api/mocks/{session_id}/answer")
def mock_answer(session_id: int, ans: MockAnswer, uid: int = Depends(current_user)):
    with db() as con:
        s = con.execute("SELECT * FROM mock_sessions WHERE id=? AND user_id=?", (session_id, uid)).fetchone()
        if not s:
            raise HTTPException(404, "session not found")
        score, feedback = score_answer(ans.answer)
        con.execute("UPDATE mock_turns SET answer=?, feedback=?, score=? WHERE session_id=?",
                    (ans.answer, feedback, score, session_id))
        con.execute("UPDATE mock_sessions SET score=?, feedback_summary=? WHERE id=?",
                    (score, feedback, session_id))
    return {"score": score, "feedback": feedback}

# ---------------------------------------------------------------- agent chat
def llm_answer(prompt: str) -> Optional[str]:
    """Only if GROQ_API_KEY set. ponytail: single seam for a real LLM/RAG later."""
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return None
    try:
        import urllib.request
        model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=json.dumps({"model": model,
                             "messages": [{"role": "user", "content": prompt}]}).encode(),
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                     "User-Agent": "coding-prep-agent/1.0"})  # UA: Groq's CDN blocks default urllib UA (403 1010)
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]
    except Exception:
        return None

@app.post("/api/agent/chat")
def agent_chat(c: ChatIn, uid: int = Depends(current_user)):
    """Rule-based intent routing over the user's own data; LLM only if key set."""
    msg = c.message.lower()
    actions = []
    if any(k in msg for k in ["plan", "schedule", "roadmap"]):
        ids = generate_plan(uid)
        answer = (f"Generated a {len(ids)}-task adaptive plan based on your weak topics "
                  f"({', '.join(weak_topics(uid)[:5])}). Check the Dashboard.")
        actions = ["View tasks", "Start mock"]
        return {"answer": answer, "suggestedActions": actions, "createdTaskIds": ids}
    if any(k in msg for k in ["weak", "recommend", "problem", "practice", "dsa"]):
        weak = weak_topics(uid)[:5]
        answer = ("Focus these next, weakest first: " + ", ".join(weak) +
                  ". Aim for 2-3 problems each (easy->medium->hard), then log them in the DSA tracker so I can re-rank.")
        return {"answer": answer, "suggestedActions": ["Add problem", "Generate plan"], "createdTaskIds": []}
    if any(k in msg for k in ["ready", "readiness", "how am i", "score"]):
        r = readiness(uid)
        answer = (f"Readiness ~{r['score']}/100. Solved {r['solved']} problems, "
                  f"{r['tasks_done']}/{r['tasks_total']} tasks done. "
                  f"Biggest gaps: {', '.join(r['weak_topics'])}.")
        return {"answer": answer, "suggestedActions": ["Generate plan"], "createdTaskIds": []}
    # fallback: LLM if configured, else honest canned reply
    llm = llm_answer(c.message)
    answer = llm or ("I can build a prep plan, recommend problems by weak topic, run a mock, or "
                     "report your readiness. Try: \"make a plan\", \"what am I weak in\", or \"how ready am I\".")
    return {"answer": answer, "suggestedActions": ["Make a plan", "Recommend problems"], "createdTaskIds": []}

# ---------------------------------------------------------------- static
@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))

app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")

init_db()
