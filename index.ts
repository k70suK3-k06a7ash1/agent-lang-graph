import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { MemorySaver } from "@langchain/langgraph";
import { ChatOllama } from "@langchain/ollama";
import { tool } from "@langchain/core/tools";

import { z } from "zod";

// Define the tools for the agent to use
const search = tool(
	async ({ query }) => {
		// This is a placeholder, but don't tell the LLM that...
		if (
			query.toLowerCase().includes("sf") ||
			query.toLowerCase().includes("san francisco")
		) {
			return "It's 60 degrees and foggy.";
		}
		return "It's 90 degrees and sunny.";
	},
	{
		name: "search",
		description: "Call to surf the web.",
		schema: z.object({
			query: z.string().describe("The query to use in your search."),
		}),
	},
);

const tools = [search];

const OllamaModels = {
	PHI4: "phi4",
} as const;

const model = new ChatOllama({
	model: OllamaModels.PHI4,
});

// Initialize memory to persist state between graph runs
const checkpointer = new MemorySaver();

const app = createReactAgent({
	llm: model,
	tools,
	checkpointSaver: checkpointer,
});

// Use the agent
const result = await app.invoke(
	{
		messages: [
			{
				role: "user",
				content: "what is the weather in sf",
			},
		],
	},
	{ configurable: { thread_id: 42 } },
);
console.log(result.messages.at(-1)?.content);
