# jobs-mcp

Portfolio site card: An MCP server that exposes live US software-engineering jobs as tools any AI client can discover and call — pulling from company ATS boards, no API keys. Stack: Python · Model Context Protocol. Published site line: MCP tools over stdio · concurrent ATS fetch · per-source failure isolation · MIT.

From the public README: jobs-mcp gives an LLM client live access to US software-engineering openings from Greenhouse, Lever, Ashby, and a community new-grad feed. Roles that require security clearance or citizenship are filtered out. Newest first. No API keys: every source is a public endpoint. Tools are search_jobs(query, location, limit) and list_sources(). A dead board is skipped. The README's own scope note: sources are a curated company list plus one community feed, not every job on the internet, and the clearance filter is lexical.

Source: https://github.com/Ashishkosana/jobs-mcp
License: MIT
