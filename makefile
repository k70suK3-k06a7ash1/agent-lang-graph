push:
	git add . && git commit -m 'chore' && git push origin
run:
	bun run agents/making-first-agent.ts
setup:
	cp .env.template .env

run-ll:
	bun run agents/low-level-implement.ts

run-research:
	bun run agents/research-company.ts