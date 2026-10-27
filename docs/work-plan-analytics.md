# Work board and event-derived analytics

Hub's work board has state columns, hierarchy, named plan/layer/performer/state
filters, a period, search and an explicit task drilldown into WorkPlanningDialog.
Preferences are scoped by project/principal. Data loads in background workers;
closing waits for active reads. State changes use the planning workflow and its
authorization, CAS and audit rules. Cards are capped at 250 per state and hierarchy
rows at 1000 to keep the view responsive; counts/tooltips disclose omitted rows.
Filters narrow the view; reports and exports retain the full selected scope.

`application/work_plan_analytics.py` calculates the report from existing work-plan
snapshot events and pipeline events. No second authoritative analytics store is
introduced. Reports identify source plan revisions/event IDs, `as_of`, period and
filters. Missing history does not become fabricated timestamps or throughput.
`infrastructure/reports/work_plans.py` defines the metric registry (localized
labels, formulas, units) used by the UI and CSV/XLSX export. Existing report
adapters remain available for other project reports.

Important aggregation rules:

* Readiness = accepted leaf items / noncancelled leaf items; an empty denominator
  is unknown. Stages are displayed but do not double-count readiness. A formally
  completed plan is counted only from its completed state.
* Reported effort is the sum of explicitly reported cumulative hours on selected
  tasks and stages. It is not elapsed wall time; projects must avoid reporting the
  same effort twice at parent and child levels.
* Lead hours = final acceptance event minus first observed creation event; cycle
  hours = final acceptance minus first recorded start event. Arithmetic means use
  currently accepted leaves with complete history and known times.
* Rework fraction = started noncancelled leaves with attempt>1 / all started
  noncancelled leaves. Rework attempts sum max(attempt-1,0).
* Acceptance throughput = transitions into accepted during the inclusive period /
  explicit period duration in calendar days. A repeated snapshot is not another
  acceptance. Reacceptance after rework is an additional acceptance event.
* Accepted frame scope counts distinct layer/frame pairs. It does not certify
  frame artifact quality or electrical acceptance.
* Workload groups open items, reported hours and remaining estimates by performer.
  Layer readiness counts matching accepted/noncancelled leaves. Scope overlaps
  are disclosed through exact item IDs; layer totals must not be summed as if
  disjoint.
* Baseline rows identify added/removed/retained scope, estimate-hour differences,
  planned-finish differences and final-acceptance minus baseline planned finish.
  Selected baseline and source revision are retained; missing estimates stay unknown.

The forecast uses explicit remaining effort, dependencies, priority, planned start
and a user-declared positive capacity in hours per **calendar day**. A performer's
tasks are serialized; independent performers may run concurrently. Blocked work,
missing estimates/assignments/capacity or unavailable dependencies yield an unknown
date with a reason. Execution-finish prediction is an estimate under these assumptions;
formal acceptance timing remains unknown. Running job counts/resource reservations
are displayed separately; measured CPU/GPU utilization stays unknown until actual
telemetry exists. Implementing that telemetry remains open.

Tests cover core event math, incomplete history, baselines/forecast, real local and
HTTP composition, migrated PostgreSQL and shown Qt board filters/drilldown/XLSX
export/close guards. See `test_work_plan_analytics*.py`, `test_work_plan_board.py`
and existing statistics regression tests for executable evidence.
