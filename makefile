push:
	git add . && git commit -m 'chore' && git push origin
run:
	bun run index.ts
setup:
	cp .env.template .env

run-ll:
	bun run low-level-implement.ts

run-research:
	bun run research-company.ts