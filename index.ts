import { MemorySaver } from "@langchain/langgraph";
import { ChatOllama } from "@langchain/ollama";
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { HumanMessage } from "@langchain/core/messages";
import { createReactAgent } from "@langchain/langgraph/prebuilt";

const agentTools = [new TavilySearchResults({ maxResults: 3 })];

const OllamaModels = {
	LLAMA3_2: "llama3.2",
} as const;

const agentModel = new ChatOllama({
	model: OllamaModels.LLAMA3_2,
});

// Initialize memory to persist state between graph runs
const agentCheckpointer = new MemorySaver();
const agent = createReactAgent({
	llm: agentModel,
	tools: agentTools,
	checkpointSaver: agentCheckpointer,
});

const agentFinalState = await agent.invoke(
	{
		messages: [
			new HumanMessage("what is the current weather in shibuya in tokyo"),
		],
	},
	{ configurable: { thread_id: "42" } },
);

console.log(
	agentFinalState.messages[agentFinalState.messages.length - 1].content,
);

const agentNextState = await agent.invoke(
	{ messages: [new HumanMessage("what about sapporo in hokkaido")] },
	{ configurable: { thread_id: "42" } },
);

console.log(
	agentNextState.messages[agentNextState.messages.length - 1].content,
);
