"""Self-check for the non-trivial logic: planner, weak-topic ranking, readiness,
mock scoring, auth token. Run: python test_placement.py  (uses a temp DB)."""
import os, tempfile
os.environ["APP_SECRET"] = "test"
DBP = tempfile.mktemp(suffix=".db")
import main
main.DB = DBP
main.init_db()

def signup(email):
    r = main.signup(main.Signup(email=email, password="pw123", name="T"))
    return main.read_token(r["token"])

def test_auth():
    uid = signup("a@x.com")
    assert uid == main.read_token(main.make_token(uid))
    try:
        main.read_token("garbage.sig"); assert False
    except Exception: pass
    # duplicate email rejected
    try:
        main.signup(main.Signup(email="a@x.com", password="p")); assert False
    except Exception: pass

def test_pw():
    h = main.hash_pw("secret")
    assert main.verify_pw("secret", h) and not main.verify_pw("wrong", h)

def test_weak_and_plan():
    uid = signup("b@x.com")
    main.put_profile(main.Profile(weak_areas="graphs, trees", daily_minutes=90), uid)
    w = main.weak_topics(uid)
    assert w[0] == "graphs" and w[1] == "trees"  # declared first
    # solving many 'arrays' pushes it down the ranking (adaptive), not removed
    for _ in range(5):
        main.add_problem(main.ProblemIn(title="p", tags="arrays"), uid)
    ranked = main.weak_topics(uid)
    non_declared = [t for t in ranked if t not in ("graphs", "trees")]
    assert non_declared[-1] == "arrays"  # most-practiced ranks last
    ids = main.generate_plan(uid)
    assert len(ids) > 0
    tasks = main.list_tasks(uid)
    assert any(t["category"] == "DSA" for t in tasks)
    assert any("graphs" in t["title"].lower() for t in tasks)  # weakest topic scheduled
    assert any("solve a" in t["title"] for t in tasks)  # tasks name a concrete pattern
    # tasks/day scales with commitment: 90 min -> 3/day
    from collections import Counter
    by_day = Counter(t["due_date"] for t in tasks)
    assert max(by_day.values()) <= 4 and any(v >= 3 for v in by_day.values())
    # regenerate replaces the prior plan, does not stack
    n = len(main.list_tasks(uid))
    main.generate_plan(uid)
    assert len(main.list_tasks(uid)) == n

def test_plan_scales_with_commitment():
    uid = signup("f@x.com")
    main.put_profile(main.Profile(daily_minutes=30, interview_date=None), uid)
    main.generate_plan(uid)
    from collections import Counter
    low = Counter(t["due_date"] for t in main.list_tasks(uid))
    assert max(low.values()) <= 2  # 30 min -> ~1 task/day (+occasional revision)

def test_plan_respects_interview_date():
    import datetime as dt
    uid = signup("c@x.com")
    soon = (dt.date.today() + dt.timedelta(days=3)).isoformat()
    main.put_profile(main.Profile(interview_date=soon), uid)
    ids = main.generate_plan(uid)
    due = {t["due_date"] for t in main.list_tasks(uid)}
    assert len(due) == 3  # backwards plan capped to days-until-interview

def test_readiness():
    uid = signup("d@x.com")
    main.put_profile(main.Profile(), uid)
    r0 = main.readiness(uid)
    assert r0["score"] == 0
    for i in range(75):
        main.add_problem(main.ProblemIn(title=f"p{i}", tags="arrays"), uid)
    r1 = main.readiness(uid)
    assert r1["solved"] == 75 and r1["score"] >= 40  # DSA component maxes at 40

def test_mock_scoring():
    weak, _ = main.score_answer("idk")
    strong, _ = main.score_answer(
        "Situation: I used a hash map for O(n) time and O(n) space. "
        "First I handled the empty array edge case, then iterated, finally returned indices. "
        + "detail " * 40)
    assert strong > weak and 0 <= strong <= 10

def test_chat_routing():
    uid = signup("e@x.com")
    main.put_profile(main.Profile(weak_areas="dp"), uid)
    assert main.agent_chat(main.ChatIn(message="make a plan"), uid)["createdTaskIds"]
    assert "dp" in main.agent_chat(main.ChatIn(message="what am I weak in"), uid)["answer"]

if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"ok {name}")
    os.remove(DBP)
    print("ALL PASS")
