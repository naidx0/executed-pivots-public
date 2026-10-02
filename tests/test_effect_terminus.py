import json

from pivots.effect.terminus import Command, is_interactive, keystrokes_to_script, parse_action


def act(**kw):
    base = {"analysis": "a", "plan": "p", "commands": [{"keystrokes": "ls\n", "duration": 0.1}]}
    base.update(kw)
    return json.dumps(base)


def test_strict_json_ok():
    r = parse_action(act())
    assert r.action and r.strict_ok and r.action.keystrokes() == ["ls\n"]


def test_think_and_fence_lenient_only():
    r = parse_action("<think>hmm {not json}</think>\n```json\n" + act() + "\n```")
    assert r.action and not r.strict_ok and r.extracted_from_noise


def test_unterminated_think_fails():
    assert parse_action("<think>still thinking " + act()).action is None


def test_schema_errors():
    assert parse_action(json.dumps({"analysis": "a", "plan": "p"})).action is None
    assert parse_action(act(task_complete="yes")).action is None
    assert parse_action(act(commands=[{"keystrokes": 3}])).action is None


def test_last_valid_object_wins():
    text = act(commands=[{"keystrokes": "a\n"}]) + " then " + act(commands=[{"keystrokes": "b\n"}])
    assert parse_action(text).action.keystrokes() == ["b\n"]


def test_keystrokes_to_script():
    s, w = keystrokes_to_script([Command("cd /app\n"), Command("C-c"), Command("echo hi"), Command(" there\n")])
    assert s == "cd /app\necho hi there\n" and not w
    s, w = keystrokes_to_script([Command("ls\n"), Command("partial")])
    assert s == "ls\n" and any("unsubmitted" in x for x in w)


def test_interactive():
    assert is_interactive("vim file\n") and is_interactive("python3\n")
    assert not is_interactive("python3 -c 'print(1)'\n") and not is_interactive("sqlite3 db -c 'x'\n")
