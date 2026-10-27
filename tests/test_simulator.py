from memoryforge.bench import get_task
from memoryforge.llm import SimulatedLLM
from memoryforge.llm import prompts as P
from memoryforge.llm.simulated import HARMFUL_TEMPLATES, LESSON_TEMPLATES


def solve_prompt(task, lessons=()):
    block = P.render_lessons([P.PromptLesson(str(i), p, t) for i, (p, t) in enumerate(lessons)])
    return P.SOLVE_USER.format(task_id=task.id, spec=task.spec, signature=task.signature,
                               entry_point=task.entry_point, lessons_block=block, reflection_block="")


def test_deterministic():
    llm, task = SimulatedLLM(), get_task("count_in_window")
    msgs = [{"role": "user", "content": solve_prompt(task)}]
    assert llm.complete(msgs, seed=3).text == llm.complete(msgs, seed=3).text


def test_matching_lesson_lowers_bug_probability():
    llm = SimulatedLLM()
    good = ("off_by_one", LESSON_TEMPLATES["off_by_one"][0])
    bad = ("off_by_one", HARMFUL_TEMPLATES["off_by_one"])
    base = llm.bug_probability("off_by_one", [], [])
    assert llm.bug_probability("off_by_one", [good], []) < base
    assert llm.bug_probability("off_by_one", [bad], []) > base
    assert llm.bug_probability("off_by_one", [("tie_ordering", "x")], []) > base  # distraction


def test_lessons_reduce_failures_over_many_seeds():
    llm, task = SimulatedLLM(), get_task("merge_inclusive_spans")
    lessons = [(p, LESSON_TEMPLATES[p][0]) for p in task.bugs]
    plain = sum(task.reference.strip() not in llm.complete([{"role": "user", "content": solve_prompt(task)}], seed=s).text
                for s in range(200))
    helped = sum(task.reference.strip() not in llm.complete([{"role": "user", "content": solve_prompt(task, lessons)}], seed=s).text
                 for s in range(200))
    assert helped < plain / 2


def test_prompt_parsing_round_trip():
    block = P.render_lessons([P.PromptLesson("7", "empty_input", "When empty, return early.")])
    assert P.parse_lessons(block) == [("empty_input", "When empty, return early.")]
    assert P.parse_distilled("pattern=off_by_one\nlesson=When x, do y.") == ("off_by_one", "When x, do y.")
