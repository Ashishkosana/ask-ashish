# tick

Portfolio site card: Durable job and cron scheduler on Postgres — many workers, one job each, and a crashed worker gets its job reclaimed. Retries with backoff, dead-letter state, recurring schedules. Stack: Python · SQLAlchemy · PostgreSQL. Published site line: SELECT ... FOR UPDATE SKIP LOCKED · 500 jobs / 10 workers · zero collisions · 14 tests.

From the public README: tick is a durable job and cron scheduler backed by a single Postgres table. Workers claim the next due job with SELECT ... FOR UPDATE SKIP LOCKED so they do not run the same job twice. A claimed job is leased for a TTL; if the worker dies, another worker reclaims it (at-least-once). A failing job retries with exponential backoff and, after max_attempts, is parked as dead. run_at in the future schedules a job; completing a recurring job enqueues its next occurrence. There is no Redis and no external lock service. SQLite is used for unit tests; Postgres is used for the SKIP LOCKED demo.

Source: https://github.com/Ashishkosana/tick
