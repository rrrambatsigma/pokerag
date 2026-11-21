export type PokemonType =
  | 'normal'
  | 'fire'
  | 'water'
  | 'electric'
  | 'grass'
  | 'ice'
  | 'fighting'
  | 'poison'
  | 'ground'
  | 'flying'
  | 'psychic'
  | 'bug'
  | 'rock'
  | 'ghost'
  | 'dragon'
  | 'dark'
  | 'steel'
  | 'fairy';

export type Strategy = 'fixed' | 'heading' | 'hierarchical';

export type Mode = 'hybrid' | 'dense' | 'bm25' | 'off';

export type Tab = 'chat' | 'settings';

export interface Settings {
  dense: boolean;
  bm25: boolean;
  strategy: Strategy;
  topK: number;
}

export interface Citation {
  n: number;
  chunkId: string;
  docId: string;
  title: string;
  dex: number;
  types: PokemonType[];
  headingPath: string[];
  excerpt: string;
  url: string;
  score: number;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  createdAt: number;
  citations?: Citation[];
  /** Settings snapshot used for this answer, kept for experiment traceability. */
  settings?: Settings;
  pending?: boolean;
}

export interface Conversation {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: Message[];
}

export interface UiState {
  version: 1;
  conversations: Conversation[];
  activeId: string | null;
  settings: Settings;
}
