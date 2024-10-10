// Please install OpenAI SDK first: `npm install openai`

import OpenAI from "openai";

const openai = new OpenAI({
        baseURL: 'https://api.deepseek.com',
        apiKey: "sk-1ffdd667c2754ccaaca2900eb1208b0d",
});

async function main(message) {
  const completion = await openai.chat.completions.create({
    messages: [{ role: "system", content: message }],
    model: "deepseek-chat",
  });

  console.log(completion.choices[0].message.content);
}

main("请进行数据分析，推荐出中国股市中你认为最具增长潜力的五只股票");