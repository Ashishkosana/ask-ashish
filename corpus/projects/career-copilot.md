# career-copilot

career-copilot is a personal job-search agent I built. It prepares one daily briefing instead of hopping between boards, mail, and a tracker. The stack named in the public README is Python, the Gmail API (OAuth), and AWS serverless (Lambda, DynamoDB, API Gateway, Cognito, CDK), with an LLM only as an option.

The live worklist is https://jobs.ashishkosana.com. The README describes it as a static page of a real run, with no API on that page.

What the README says it does, without adding counts: it reads public ATS job APIs (no scraper and no vendor key on that path), drops roles that fail an eligibility gate (including clearance and sponsorship limits) instead of scoring them, scores what remains against the résumé as a set of covered and missing requirements, remembers postings between runs, and can triage Gmail. Reply drafts are Gmail drafts a person reviews. They are never auto-sent. The project does not auto-submit applications. POST /applied records that the person applied and returns submitted false.

This note does not copy point-in-time posting totals from the README. Ask the repo if you need those figures.

Source: https://github.com/Ashishkosana/career-copilot
