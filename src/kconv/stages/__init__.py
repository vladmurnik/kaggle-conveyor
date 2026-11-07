"""Pipeline stages.

Each stage exposes ``run_<stage>(ctx: RunContext) -> dict`` and writes
artifacts under ``ctx.run_dir``.  The orchestrator owns ordering and state.
"""
