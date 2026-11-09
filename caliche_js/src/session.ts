/**
 * Agent session management for conversation state and forking.
 */

import type { MessageJs, SharedState } from "./index";
import { native } from "./native/module";

/** Native session handle methods used by the TypeScript wrapper. */
interface NativeAgentSessionHandle {
  readonly id: string;
  readonly sharedState?: SharedState;
  readonly shared_state?: SharedState;
  fork?(newId: string): Promise<NativeAgentSessionHandle>;
  forkSession?(newId: string): Promise<NativeAgentSessionHandle>;
  merge?(other: NativeAgentSessionHandle): Promise<void>;
  mergeSession?(other: NativeAgentSessionHandle): Promise<void>;
  getHistory?(): MessageJs[];
  get_history?(): MessageJs[];
  addMessage?(message: MessageJs): void;
  add_message?(message: MessageJs): void;
  clearHistory?(): void;
  clear_history?(): void;
}

/**
 * Agent session for managing conversation state and enabling session forking.
 */
export class AgentSession {
  private _session: NativeAgentSessionHandle;

  constructor(id: string, history?: readonly MessageJs[], sharedState?: SharedState) {
    if (sharedState) {
      this._session = new native.AgentSession(
        id,
        undefined,
        sharedState
      ) as NativeAgentSessionHandle;
    } else if (history) {
      if (typeof native.AgentSession.with_history === "function") {
        this._session = native.AgentSession.with_history(id, history) as NativeAgentSessionHandle;
      } else if (typeof native.AgentSession.withHistory === "function") {
        this._session = native.AgentSession.withHistory(id, history) as NativeAgentSessionHandle;
      } else {
        this._session = new native.AgentSession(id, history) as NativeAgentSessionHandle;
      }
    } else {
      this._session = new native.AgentSession(id) as NativeAgentSessionHandle;
    }
  }

  /** Create a session with existing conversation history. */
  static withHistory(id: string, history: readonly MessageJs[]): AgentSession {
    return new AgentSession(id, history);
  }

  /** Session id. */
  get id(): string {
    return this._session.id;
  }

  /**
   * Fork this session into a new isolated session.
   *
   * The forked session has a new id, a copy of history, and the same shared state.
   */
  async fork(newId: string): Promise<AgentSession> {
    const forked =
      typeof this._session.fork === "function"
        ? await this._session.fork(newId)
        : await this._session.forkSession?.(newId);
    if (!forked) {
      throw new Error("Native AgentSession.fork is not available");
    }
    const session = Object.create(AgentSession.prototype) as AgentSession;
    session._session = forked;
    return session;
  }

  /** Append another session's conversation history into this session. */
  async merge(other: AgentSession): Promise<void> {
    const otherSession = other._session;
    if (typeof this._session.merge === "function") {
      await this._session.merge(otherSession);
    } else if (typeof this._session.mergeSession === "function") {
      await this._session.mergeSession(otherSession);
    } else {
      throw new Error("Native AgentSession.merge is not available");
    }
  }

  /** Current conversation history. */
  getHistory(): MessageJs[] {
    if (typeof this._session.getHistory === "function") {
      return this._session.getHistory();
    }
    if (typeof this._session.get_history === "function") {
      return this._session.get_history();
    }
    return [];
  }

  /** Append a message to the conversation history. */
  addMessage(message: MessageJs): void {
    if (typeof this._session.addMessage === "function") {
      this._session.addMessage(message);
      return;
    }
    if (typeof this._session.add_message === "function") {
      this._session.add_message(message);
    }
  }

  /** Clear the conversation history. */
  clearHistory(): void {
    if (typeof this._session.clearHistory === "function") {
      this._session.clearHistory();
      return;
    }
    if (typeof this._session.clear_history === "function") {
      this._session.clear_history();
    }
  }

  /** Shared state handle for cross-session coordination. */
  get sharedState(): SharedState {
    if (this._session.sharedState !== undefined) {
      return this._session.sharedState;
    }
    if (this._session.shared_state !== undefined) {
      return this._session.shared_state;
    }
    throw new Error("Native AgentSession.sharedState is not available");
  }
}
