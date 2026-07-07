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
# topic -> concrete patterns, so tasks name a real thing to solve (not "do graphs")
TOPIC_PATTERNS = {
    "arrays": ["two-sum via hash map", "sliding window max/min", "prefix sums", "Kadane's max subarray", "Dutch-flag partition"],
    "strings": ["anagram groups (freq map)", "longest palindromic substring", "min-window substring", "string-to-int parsing"],
    "hashing": ["subarray sum = k", "longest consecutive sequence", "group by key", "first unique char"],
    "two pointers": ["3-sum", "container with most water", "remove duplicates in place", "trapping rain water"],
    "binary search": ["first/last occurrence", "search in rotated array", "median of two arrays", "binary search on the answer"],
    "linked list": ["reverse a list", "detect cycle (Floyd)", "merge k sorted lists", "remove nth from end"],
    "stacks": ["valid parentheses", "next greater element", "min stack", "largest rectangle in histogram"],
    "recursion": ["subsets", "permutations", "combination sum", "N-queens"],
    "trees": ["level-order traversal", "lowest common ancestor", "validate BST", "diameter of tree", "serialize/deserialize"],
    "graphs": ["BFS shortest path", "DFS connected components", "topological sort", "Dijkstra", "Union-Find / DSU"],
    "dp": ["0/1 knapsack", "longest common subsequence", "coin change", "longest increasing subsequence", "edit distance"],
    "greedy": ["interval scheduling", "jump game", "gas station", "task scheduler"],
}
DSA_TOPICS = list(TOPIC_PATTERNS.keys())
CS_TOPICS = [
    "OS: processes vs threads, scheduling, deadlock",
    "DBMS: normalization, indexing, transactions & ACID",
    "CN: TCP vs UDP, HTTP, DNS, the TLS handshake",
    "OOP: SOLID, composition vs inheritance, common patterns",
    "System design: caching, load balancing, DB sharding",
]
BEHAVIORAL = [
    "STAR: a project you're proud of",
    "STAR: a conflict you resolved on a team",
    "STAR: a failure and what you learned",
    "STAR: a time you took ownership / led",
    "'Tell me about yourself' — a 2-minute pitch",
]
MOCK_TYPES = ["DSA", "behavioral", "project", "system design basics"]


def weak_topics(uid: int) -> list[str]:
    """Adaptive ranking: declared weak first, then DSA topics by fewest solved
    problems (least-practiced = weakest). Shifts as the user logs problems."""
    with db() as con:
        prof = con.execute("SELECT weak_areas FROM profiles WHERE user_id=?", (uid,)).fetchone()
        solved_tags = con.execute(
            "SELECT tags FROM problems WHERE user_id=? AND status='solved'", (uid,)).fetchall()
    counts = {t: 0 for t in DSA_TOPICS}
    for r in solved_tags:
        for t in (r["tags"] or "").split(","):
            t = t.strip()
            if t in counts:
                counts[t] += 1
    declared = [w.strip().lower() for w in (prof["weak_areas"] if prof else "").split(",") if w.strip()]
    ranked = sorted(DSA_TOPICS, key=lambda t: counts[t])  # fewest solved first
    return list(dict.fromkeys(declared + ranked))

def _weakest_mock_type(uid: int) -> str:
    """Mock type with the lowest average score (untried types win first)."""
    with db() as con:
        rows = con.execute(
            "SELECT type, AVG(score) a, COUNT(*) c FROM mock_sessions WHERE user_id=? GROUP BY type",
            (uid,)).fetchall()
    avg = {r["type"]: r["a"] for r in rows if r["a"] is not None}
    untried = [t for t in MOCK_TYPES if t not in avg]
    if untried:
        return untried[0]
    return min(avg, key=avg.get)

def generate_plan(uid: int) -> list[dict]:
    with db() as con:
        prof = con.execute("SELECT * FROM profiles WHERE user_id=?", (uid,)).fetchone()
    if not prof:
        raise HTTPException(400, "create a profile first")
    weak = weak_topics(uid)
    days = 14
    if prof["interview_date"]:
        try:
            d = (dt.date.fromisoformat(prof["interview_date"]) - dt.date.today()).days
            days = max(1, min(d, 30))
        except ValueError:
            pass
    # tasks/day scale with the user's stated commitment (~30 min per task), 1..4
    per_day = max(1, min((prof["daily_minutes"] or 60) // 30, 4))
    # concrete DSA queue: each weak topic expanded into its named patterns
    dsa_queue = [(topic, pat) for topic in weak for pat in TOPIC_PATTERNS.get(topic, [f"core {topic}"])]
    mock_type = _weakest_mock_type(uid)

    tasks, today, dsa_i = [], dt.date.today(), 0
    for d in range(days):
        due = (today + dt.timedelta(days=d)).isoformat()
        back_half = d >= days * 0.5
        slots = per_day
        # mocks: back half only, escalating toward the interview, weakest type first
        if back_half and d % 2 == days % 2 and slots > 0:
            tasks.append(("Mock", f"Timed mock: {mock_type}", "high", due)); slots -= 1
            mock_type = MOCK_TYPES[(MOCK_TYPES.index(mock_type) + 1) % len(MOCK_TYPES)]
        # one CS topic every other day (front-loaded learning)
        if d % 2 == 0 and slots > 0:
            tasks.append(("CS", "Revise " + CS_TOPICS[(d // 2) % len(CS_TOPICS)], "medium", due)); slots -= 1
        # one behavioral prompt every third day
        if d % 3 == 2 and slots > 0:
            tasks.append(("Behavioral", BEHAVIORAL[(d // 3) % len(BEHAVIORAL)], "low", due)); slots -= 1
        # fill remaining slots with concrete, non-repeating DSA patterns
        while slots > 0 and dsa_queue:
            topic, pat = dsa_queue[dsa_i % len(dsa_queue)]; dsa_i += 1; slots -= 1
            tasks.append(("DSA", f"{topic.title()} - solve a '{pat}' problem",
                          "high" if not back_half else "medium", due))
        # spaced revision: in the back third, re-do an earlier pattern from memory
        if back_half and dsa_i > 4 and d % 3 == 0:
            topic, pat = dsa_queue[(dsa_i - 4) % len(dsa_queue)]
            tasks.append(("Revision", f"Redo '{pat}' ({topic}) from memory, no hints", "medium", due))

    now = dt.datetime.utcnow().isoformat()
    ids = []
    with db() as con:
        # regenerate replaces the prior planner plan instead of stacking on it
        con.execute("DELETE FROM tasks WHERE user_id=? AND source_agent='planner'", (uid,))
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
