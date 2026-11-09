export interface SseEvent {
  readonly id?: string;
  readonly event?: string;
  readonly data: string;
  readonly retry?: number;
}

function parseSseEventBlock(block: string): SseEvent | null {
  const trimmed = block.trim();
  if (trimmed.length === 0) {
    return null;
  }

  let id: string | undefined;
  let event: string | undefined;
  let retry: number | undefined;
  const dataLines: string[] = [];

  for (const line of trimmed.split("\n")) {
    if (line.startsWith(":")) {
      continue;
    }
    const separator = line.indexOf(":");
    const field = separator === -1 ? line : line.slice(0, separator);
    const value = separator === -1 ? "" : line.slice(separator + 1).replace(/^\s/, "");

    switch (field) {
      case "id":
        id = value;
        break;
      case "event":
        event = value;
        break;
      case "retry":
        retry = Number.parseInt(value, 10);
        break;
      case "data":
        dataLines.push(value);
        break;
      default:
        break;
    }
  }

  if (dataLines.length === 0) {
    return null;
  }

  return {
    ...(id !== undefined ? { id } : {}),
    ...(event !== undefined ? { event } : {}),
    ...(retry !== undefined && !Number.isNaN(retry) ? { retry } : {}),
    data: dataLines.join("\n"),
  };
}

/**
 * Decode one or more SSE events from a text chunk.
 */
export function decodeSseChunk(chunk: string): SseEvent[] {
  return chunk
    .split(/\n\n|\r\n\r\n/)
    .map(parseSseEventBlock)
    .filter((event): event is SseEvent => event !== null);
}

/**
 * Decode a stream of SSE chunks into discrete events.
 */
export async function* decodeSseStream(
  stream: AsyncIterable<string> | Iterable<string>
): AsyncGenerator<SseEvent, void, unknown> {
  let buffer = "";
  for await (const chunk of stream) {
    buffer += chunk;
    const parts = buffer.split(/\n\n|\r\n\r\n/);
    buffer = parts.pop() ?? "";
    for (const part of parts) {
      const event = parseSseEventBlock(part);
      if (event) {
        yield event;
      }
    }
  }

  const trailing = parseSseEventBlock(buffer);
  if (trailing) {
    yield trailing;
  }
}

/**
 * Encode a single SSE event block.
 */
export function encodeSseEvent(event: SseEvent): string {
  const lines: string[] = [];
  if (event.id !== undefined) {
    lines.push(`id: ${event.id}`);
  }
  if (event.event !== undefined) {
    lines.push(`event: ${event.event}`);
  }
  if (event.retry !== undefined) {
    lines.push(`retry: ${event.retry}`);
  }
  for (const line of event.data.split("\n")) {
    lines.push(`data: ${line}`);
  }
  lines.push("");
  lines.push("");
  return lines.join("\n");
}
