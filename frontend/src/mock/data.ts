import type { Citation, Conversation, Settings } from '../types';

export const SNAPSHOT = {
  name: 'pokerag-corpus-v2',
  docs: 386,
  indexVersion: 'fa5a8dafad1e',
  model: 'bge-m3',
  dimension: 1024,
} as const;

export const DEFAULT_SETTINGS: Settings = {
  dense: true,
  bm25: true,
  strategy: 'heading',
  topK: 5,
};

export function wikiUrl(title: string): string {
  return `https://bulbapedia.bulbagarden.net/wiki/${encodeURIComponent(title.replaceAll(' ', '_'))}`;
}

function cite(
  n: number,
  spec: Omit<Citation, 'n' | 'url'> & { url?: string },
): Citation {
  return {
    n,
    url: spec.url ?? wikiUrl(spec.title),
    chunkId: spec.chunkId,
    docId: spec.docId,
    title: spec.title,
    dex: spec.dex,
    types: spec.types,
    headingPath: spec.headingPath,
    excerpt: spec.excerpt,
    score: spec.score,
  };
}

const BIOLOGY_CHARIZARD = cite(1, {
  chunkId: '4f2a91c0d7e3b861',
  docId: 'bulba:0002363',
  title: 'Charizard (Pokémon)',
  dex: 6,
  types: ['fire', 'flying'],
  headingPath: ['Biology'],
  excerpt:
    'Charizard is a draconic, bipedal Pokémon. It is primarily orange with a cream underside from the chest to the tip of its tail. It has two wings with teal undersides.',
  score: 0.92,
});

const ABILITY_CHARIZARD = cite(2, {
  chunkId: '91c04e2a7b6d3f08',
  docId: 'bulba:0002363',
  title: 'Charizard (Pokémon)',
  dex: 6,
  types: ['fire', 'flying'],
  headingPath: ['Game data', 'Ability'],
  excerpt:
    'Blaze powers up Fire-type moves when the Pokémon has less than one third of its HP. Solar Power raises Special Attack in harsh sunlight but costs HP each turn.',
  score: 0.74,
});

const BIOLOGY_EEVEE = cite(1, {
  chunkId: 'c8120e5f9a4d7b36',
  docId: 'bulba:0002361',
  title: 'Eevee (Pokémon)',
  dex: 133,
  types: ['normal'],
  headingPath: ['Biology'],
  excerpt:
    'Eevee evolves into one of eight different Pokémon: Vaporeon with a Water Stone, Jolteon with a Thunder Stone, Flareon with a Fire Stone, and further forms through friendship, location, or time of day.',
  score: 0.9,
});

const BIOLOGY_ZAPDOS = cite(1, {
  chunkId: 'a1742f8c33d90be5',
  docId: 'bulba:0002358',
  title: 'Zapdos (Pokémon)',
  dex: 145,
  types: ['fighting', 'flying'],
  headingPath: ['Biology'],
  excerpt:
    'Galarian Zapdos is a dual-type Fighting / Flying regional form introduced in The Crown Tundra. It has a muscular body and runs across the land at high speed.',
  score: 0.88,
});

const BIOLOGY_VENUSAUR = cite(1, {
  chunkId: '5d90b1e7c24a6f83',
  docId: 'bulba:0001538',
  title: 'Venusaur (Pokémon)',
  dex: 3,
  types: ['grass', 'poison'],
  headingPath: ['Biology'],
  excerpt:
    'Venusaur can Mega Evolve into Mega Venusaur using the Venusaurite. It evolves from Ivysaur starting at level 32 and is the final form of Bulbasaur.',
  score: 0.86,
});

interface MockAnswer {
  text: string;
  citations: Citation[];
}

export function answerFor(query: string): MockAnswer {
  const q = query.toLowerCase();

  if (/(charizard|charmander|flamethrower|blaze)/.test(q)) {
    return {
      text:
        'Charizard has the ability Blaze, and Solar Power as its hidden ability. It is a dual-type Fire / Flying Pokémon introduced in Generation I and the final form of Charmander.',
      citations: [BIOLOGY_CHARIZARD, ABILITY_CHARIZARD],
    };
  }

  if (/(eevee|evolusi|evolve|evolution)/.test(q)) {
    return {
      text:
        'Eevee is a Normal-type Pokémon that can evolve into eight different Pokémon. The first three depend on a stone: Vaporeon, Jolteon, and Flareon.',
      citations: [BIOLOGY_EEVEE],
    };
  }

  if (/(zapdos|galar|galarian)/.test(q)) {
    return {
      text:
        'Galarian Zapdos is a dual-type Fighting / Flying Pokémon introduced in The Crown Tundra. The Kantonian form is Electric / Flying.',
      citations: [BIOLOGY_ZAPDOS],
    };
  }

  if (/(venusaur|mega|bulbasaur)/.test(q)) {
    return {
      text:
        'Venusaur can Mega Evolve into Mega Venusaur using the Venusaurite. It evolves from Ivysaur at level 32 and is the final form of Bulbasaur.',
      citations: [BIOLOGY_VENUSAUR],
    };
  }

  return {
    text:
      'Demo ini belum memiliki jawaban untuk pertanyaan tersebut. Coba tanyakan tentang Charizard, Eevee, Zapdos, atau Venusaur.',
    citations: [],
  };
}

export const SUGGESTIONS = [
  'Kemampuan Charizard',
  'Evolusi Eevee',
  'Tipe Galarian Zapdos',
];

export function seedConversations(): Conversation[] {
  const now = Date.now();
  const hour = 3_600_000;
  const day = 24 * hour;

  return [
    {
      id: 'seed-charizard',
      title: 'Kemampuan Charizard',
      createdAt: now - hour,
      updatedAt: now - hour,
      messages: [
        {
          id: 'seed-charizard-u1',
          role: 'user',
          text: 'Which ability does Charizard have?',
          createdAt: now - hour,
        },
        {
          id: 'seed-charizard-a1',
          role: 'assistant',
          text: 'Charizard has the ability Blaze, and Solar Power as its hidden ability. It is a dual-type Fire / Flying Pokémon introduced in Generation I.',
          createdAt: now - hour + 900,
          citations: [BIOLOGY_CHARIZARD, ABILITY_CHARIZARD],
          settings: DEFAULT_SETTINGS,
        },
      ],
    },
    {
      id: 'seed-zapdos',
      title: 'Tipe Galarian Zapdos',
      createdAt: now - day,
      updatedAt: now - day,
      messages: [
        {
          id: 'seed-zapdos-u1',
          role: 'user',
          text: 'What type is Galarian Zapdos?',
          createdAt: now - day,
        },
        {
          id: 'seed-zapdos-a1',
          role: 'assistant',
          text: 'Galarian Zapdos is a dual-type Fighting / Flying Pokémon introduced in The Crown Tundra.',
          createdAt: now - day + 800,
          citations: [BIOLOGY_ZAPDOS],
          settings: { ...DEFAULT_SETTINGS, strategy: 'fixed' },
        },
      ],
    },
    {
      id: 'seed-eevee',
      title: 'Evolusi Eevee',
      createdAt: now - 2 * day,
      updatedAt: now - 2 * day,
      messages: [
        {
          id: 'seed-eevee-u1',
          role: 'user',
          text: 'How does Eevee evolve?',
          createdAt: now - 2 * day,
        },
        {
          id: 'seed-eevee-a1',
          role: 'assistant',
          text: 'Eevee is a Normal-type Pokémon that can evolve into eight different Pokémon, starting with Vaporeon, Jolteon, and Flareon.',
          createdAt: now - 2 * day + 700,
          citations: [BIOLOGY_EEVEE],
          settings: { ...DEFAULT_SETTINGS, bm25: false },
        },
      ],
    },
  ];
}
