import OpenAI from "openai";
const openai = new OpenAI({
	apiKey: process.env.OPENAI_API_KEY,
});
const completion = await openai.chat.completions.create({
	model: "gpt-3.5-turbo",
	store: true,
	messages: [{ role: "user", content: "write a haiku about ai" }],
});

console.log({ completion });
