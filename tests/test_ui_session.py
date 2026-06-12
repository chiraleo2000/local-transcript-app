"""Per-tab UI session registry."""

from backend.ui_session import (
    DEFAULT_TAB_ID,
    fresh_cancel_event,
    init_tab_instance_id,
    resolve_runtime,
)


def test_tabs_have_isolated_cancel_events():
    runtime_a, tab_a = resolve_runtime("tab-a")
    runtime_b, tab_b = resolve_runtime("tab-b")
    assert tab_a != tab_b
    event_a = fresh_cancel_event(runtime_a)
    assert runtime_b["cancel_event"] is not event_a
    assert not runtime_b["cancel_event"].is_set()


def test_empty_tab_id_uses_stable_default():
    _, tid_a = resolve_runtime("")
    _, tid_b = resolve_runtime(None)
    assert tid_a == DEFAULT_TAB_ID
    assert tid_b == DEFAULT_TAB_ID


def test_init_tab_instance_id_preserves_existing():
    assert init_tab_instance_id("my-tab") == "my-tab"


def test_init_tab_instance_id_generates_when_empty():
    new_id = init_tab_instance_id("")
    assert new_id
    assert new_id != DEFAULT_TAB_ID


def test_active_job_tracking():
    from backend.ui_session import clear_active_job, is_job_running, set_active_job

    runtime, _ = resolve_runtime("job-tab")
    assert not is_job_running(runtime)
    set_active_job(runtime, "job-123", None)
    runtime["progress"].start("job-123")
    assert is_job_running(runtime)
    clear_active_job(runtime)
    runtime["progress"].reset()
    assert not is_job_running(runtime)
