import type { Citation, Message, Settings } from '../types';

interface ChatPayload {
  message: Message;
}

export interface AssistantAnswer {
  text: string;
  citations: Citation[];
}

/** Send one question to the FastAPI backend through the Vite proxy. */
export async function askApi(
  text: string,
  settings: Settings,
): Promise<AssistantAnswer> {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: text, settings }),
  });

  if (!response.ok) {
    throw new Error(`API ${response.status}`);
  }

  const payload = (await response.json()) as ChatPayload;
  return {
    text: payload.message.text,
    citations: payload.message.citations ?? [],
  };
}
