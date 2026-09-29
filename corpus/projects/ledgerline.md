# ledgerline

Portfolio site card: Exactly-once payments — client retries cannot slip a second charge through. Storage-layer idempotency over a payment state machine, a double-entry ledger with automated reconciliation, and a transactional outbox with a dead-letter queue. Stack: Python · FastAPI · PostgreSQL · Docker. Published site line: 0 double-charges / 200 concurrent · ~1,460 tx/sec · 27 tests.

From the public README: ledgerline is a payments backend that stays correct under failure. A payment moves created → authorized → captured → refunded (or failed), and illegal jumps are rejected. An idempotency key is enforced with a database UNIQUE constraint, so a retry returns the original payment. Capture and refund post balanced debit and credit rows in the same transaction, and reconcile() checks that the ledger sums to zero. Events are written in that same transaction (a transactional outbox); a relay publishes them; a consumer dedups redeliveries; poison messages land in a dead-letter queue after repeated attempts.

Source: https://github.com/Ashishkosana/ledgerline
