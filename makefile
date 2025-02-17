push:
	git add . && git commit -m 'chore' && git push origin
run:
	bun run agents/making-first-agent.ts

multi-agent:
	bun run agents/multi-agent.ts
setup:
	cp .env.template .env


openai-agent:
	bun run agents/openai-agent.ts

run-ll:
	bun run agents/low-level-implement.ts

run-research:
	bun run agents/research-company.ts
test:
	bun run llm/openai.ts