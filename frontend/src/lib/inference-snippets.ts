import { config } from "@/config";
import type { DecisionInputRequest } from "@/openapi";

export function inferenceBaseUrl(apiUrl = config.apiUrl): string {
  return `${apiUrl.replace(/\/$/, "")}/api/v1`;
}

function choiceCriteria(decision: DecisionInputRequest): Record<string, string | null> {
  const described = new Map<string, string>();
  for (const line of decision.question.split("\n").slice(1)) {
    const split = line.indexOf(": ");
    if (split <= 0) continue;
    const name = line.slice(0, split);
    const description = line.slice(split + 2).trim();
    if (decision.options.includes(name) && description) described.set(name, description);
  }
  return Object.fromEntries(
    decision.options.map((option) => [option, described.get(option) ?? null])
  );
}

export function buildDecisionSnippets(
  modelId: string,
  decision: DecisionInputRequest,
  apiUrl = config.apiUrl
) {
  const endpoint = `${inferenceBaseUrl(apiUrl)}/systemone`;
  const criteria =
    decision.kind === "noul"
      ? {
          false: decision.options[0] ?? "false",
          true: decision.options[1] ?? "true",
        }
      : decision.kind === "score"
        ? decision.options
        : choiceCriteria(decision);
  const body = JSON.stringify(
    {
      model: modelId,
      questions: {
        decision: {
          criteria,
          instructions: decision.question,
          type: decision.kind,
        },
      },
      state: decision.state,
    },
    null,
    2
  );
  const shellQuote = (value: string) => `'${value.replaceAll("'", "'\"'\"'")}'`;
  return {
    curl: [
      `curl --fail-with-body --silent --show-error --max-time 200 ${shellQuote(endpoint)} \\`,
      '  -H "Authorization: Bearer $OVERMIND_API_KEY" \\',
      '  -H "Content-Type: application/json" \\',
      `  --data ${shellQuote(body)}`,
    ].join("\n"),
    python: [
      "import os",
      "",
      "import requests",
      "",
      "# Example from this model's training version. Replace state with your input.",
      "response = requests.post(",
      `    ${JSON.stringify(endpoint)},`,
      '    headers={"Authorization": f"Bearer {os.environ[\'OVERMIND_API_KEY\']}", "Content-Type": "application/json"},',
      `    data=r'''${body}''',`,
      "    timeout=200,",
      ")",
      "response.raise_for_status()",
      'answer = response.json()["answers"]["decision"]',
      decision.kind === "noul"
        ? 'print(answer["noul"])'
        : decision.kind === "score"
          ? 'print(answer["score"])\nprint(answer["probabilities"])'
          : 'print(answer["choice"])\nprint(answer["probabilities"])',
    ].join("\n"),
  };
}

export function buildCurlSnippet({
  baseUrl,
  modelId,
}: {
  baseUrl: string;
  modelId: string;
}): string {
  return [
    `curl ${baseUrl}/chat/completions \\`,
    `  -H "Authorization: Bearer $OVERMIND_API_KEY" \\`,
    `  -H "Content-Type: application/json" \\`,
    `  -d '{`,
    `    "model": "${modelId}",`,
    `    "messages": [{"role": "user", "content": "Hello"}],`,
    `    "max_tokens": 256`,
    `  }'`,
  ].join("\n");
}

export function buildPythonSnippet({
  baseUrl,
  modelId,
}: {
  baseUrl: string;
  modelId: string;
}): string {
  return [
    `import os`,
    `from openai import OpenAI`,
    ``,
    `client = OpenAI(`,
    `    base_url="${baseUrl}",`,
    `    api_key=os.environ["OVERMIND_API_KEY"],`,
    `)`,
    ``,
    `resp = client.chat.completions.create(`,
    `    model="${modelId}",`,
    `    messages=[{"role": "user", "content": "Hello"}],`,
    `    max_tokens=256,`,
    `)`,
    `print(resp.choices[0].message.content)`,
  ].join("\n");
}
