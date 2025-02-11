import { type BaseMessage, HumanMessage } from "@langchain/core/messages";
import { SystemMessage } from "@langchain/core/messages";
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { tool } from "@langchain/core/tools";
import { Annotation } from "@langchain/langgraph";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { ChatOllama } from "@langchain/ollama";
import { z } from "zod";
import { JsonOutputToolsParser } from "langchain/output_parsers";
import {
	ChatPromptTemplate,
	MessagesPlaceholder,
} from "@langchain/core/prompts";
import type { Runnable } from "@langchain/core/runnables";
import type { StructuredToolInterface } from "@langchain/core/tools";
import type { MessagesAnnotation } from "@langchain/langgraph";
const ResearchTeamState = Annotation.Root({
	messages: Annotation<BaseMessage[]>({
		reducer: (x, y) => x.concat(y),
	}),
	team_members: Annotation<string[]>({
		reducer: (x, y) => x.concat(y),
	}),
	next: Annotation<string>({
		reducer: (x, y) => y ?? x,
		default: () => "supervisor",
	}),
	instructions: Annotation<string>({
		reducer: (x, y) => y ?? x,
		default: () => "Solve the human's question.",
	}),
});

const OllamaModels = {
	LLAMA3_2: "llama3.2",
} as const;

const llm = new ChatOllama({
	model: OllamaModels.LLAMA3_2,
});

const tavilyTool = new TavilySearchResults();

async function runAgentNode(params: {
	// biome-ignore lint/suspicious/noExplicitAny: <explanation>
	state: any;
	agent: Runnable;
	name: string;
}) {
	const { state, agent, name } = params;
	const result = await agent.invoke({
		messages: state.messages,
	});
	const lastMessage = result.messages[result.messages.length - 1];
	return {
		messages: [new HumanMessage({ content: lastMessage.content, name })],
	};
}

const scrapeWebpage = tool(
	async (input) => {
		const loader = new CheerioWebBaseLoader(input.url);
		const docs = await loader.load();
		const formattedDocs = docs.map(
			(doc) =>
				`<Document name="${doc.metadata?.title}">\n${doc.pageContent}\n</Document>`,
		);
		return formattedDocs.join("\n\n");
	},
	{
		name: "scrape_webpage",
		description: "Scrape the contents of a webpage.",
		schema: z.object({
			url: z.string(),
		}),
	},
);

const agentStateModifier = (
	systemPrompt: string,
	tools: StructuredToolInterface[],
	teamMembers: string[],
): ((state: typeof MessagesAnnotation.State) => BaseMessage[]) => {
	const toolNames = tools.map((t) => t.name).join(", ");
	const systemMsgStart = new SystemMessage(
		`${systemPrompt}\nWork autonomously according to your specialty, using the tools available to you. Do not ask for clarification. Your other team members (and other teams) will collaborate with you with their own specialties. You are chosen for a reason! You are one of the following team members: ${teamMembers.join(", ")}.`,
	);
	const systemMsgEnd = new SystemMessage(
		`Supervisor instructions: ${systemPrompt}\nRemember, you individually can only use these tools: ${toolNames}\n\nEnd if you have already completed the requested task. Communicate the work completed.`,
	);

	// biome-ignore lint/suspicious/noExplicitAny: <explanation>
	return (state: typeof MessagesAnnotation.State): any[] => [
		systemMsgStart,
		...state.messages,
		systemMsgEnd,
	];
};

const searchNode = (state: typeof ResearchTeamState.State) => {
	const stateModifier = agentStateModifier(
		"You are a research assistant who can search for up-to-date info using the tavily search engine.",
		[tavilyTool],
		state.team_members ?? ["Search"],
	);
	const searchAgent = createReactAgent({
		llm,
		tools: [tavilyTool],
		stateModifier,
	});
	return runAgentNode({ state, agent: searchAgent, name: "Search" });
};

const researchNode = (state: typeof ResearchTeamState.State) => {
	const stateModifier = agentStateModifier(
		"You are a research assistant who can scrape specified urls for more detailed information using the scrapeWebpage function.",
		[scrapeWebpage],
		state.team_members ?? ["WebScraper"],
	);
	const researchAgent = createReactAgent({
		llm,
		tools: [scrapeWebpage],
		stateModifier,
	});
	return runAgentNode({ state, agent: researchAgent, name: "WebScraper" });
};

async function createTeamSupervisor(
	llm: ChatOllama,
	systemPrompt: string,
	members: string[],
): Promise<Runnable> {
	const options = ["FINISH", ...members];
	const routeTool = {
		name: "route",
		description: "Select the next role.",
		schema: z.object({
			reasoning: z.string(),
			next: z.enum(["FINISH", ...members]),
			instructions: z
				.string()
				.describe(
					"The specific instructions of the sub-task the next role should accomplish.",
				),
		}),
	};
	let prompt = ChatPromptTemplate.fromMessages([
		["system", systemPrompt],
		new MessagesPlaceholder("messages"),
		[
			"system",
			"Given the conversation above, who should act next? Or should we FINISH? Select one of: {options}",
		],
	]);
	prompt = await prompt.partial({
		options: options.join(", "),
		team_members: members.join(", "),
	});

	const supervisor = prompt
		.pipe(
			llm.bindTools([routeTool], {
				tool_choice: "route",
			}),
		)
		.pipe(new JsonOutputToolsParser())
		// select the first one
		.pipe((x) => ({
			next: x[0].args.next,
			instructions: x[0].args.instructions,
		}));

	return supervisor;
}

const supervisorAgent = await createTeamSupervisor(
	llm,
	"You are a supervisor tasked with managing a conversation between the" +
		" following workers:  {team_members}. Given the following user request," +
		" respond with the worker to act next. Each worker will perform a" +
		" task and respond with their results and status. When finished," +
		" respond with FINISH.\n\n" +
		" Select strategically to minimize the number of steps taken.",
	["Search", "WebScraper"],
);

import { END, START, StateGraph } from "@langchain/langgraph";

const researchGraph = new StateGraph(ResearchTeamState)
	.addNode("Search", searchNode)
	.addNode("supervisor", supervisorAgent)
	.addNode("WebScraper", researchNode)
	// Define the control flow
	.addEdge("Search", "supervisor")
	.addEdge("WebScraper", "supervisor")
	.addConditionalEdges("supervisor", (x) => x.next, {
		Search: "Search",
		WebScraper: "WebScraper",
		FINISH: END,
	})
	.addEdge(START, "supervisor");

const researchChain = researchGraph.compile();

const streamResults = researchChain.stream(
	{
		messages: [new HumanMessage("What's the price of a big mac in Argentina?")],
	},
	{ recursionLimit: 100 },
);

for await (const output of await streamResults) {
	if (!output?.__end__) {
		console.log(output);
		console.log("----");
	}
}
