import { MODE_LABEL, modeFromSettings } from '../lib/mode';
import { SNAPSHOT } from '../mock/data';
import type { Settings, Strategy } from '../types';
import { PixelChip, PixelStepper, PixelSwitch } from './ui';

interface SettingsPageProps {
  settings: Settings;
  onChange: (patch: Partial<Settings>) => void;
}

const STRATEGIES: { value: Strategy; label: string }[] = [
  { value: 'fixed', label: 'FIXED' },
  { value: 'heading', label: 'HEADING' },
  { value: 'hierarchical', label: 'HIERARCHICAL' },
];

export function SettingsPage({ settings, onChange }: SettingsPageProps) {
  const mode = modeFromSettings(settings);

  return (
    <div className="win settings-shell">
      <div className="win-title">
        <span>SETELAN</span>
        <span />
      </div>

      <div className="settings">
        <div className="card blue">
          <h3>PENCARIAN</h3>
          <div className="body">
            <div className="row">
              <span>
                Makna teks
                <span className="sub">dense · Qdrant</span>
              </span>
              <PixelSwitch
                checked={settings.dense}
                label="Pencarian makna teks"
                onChange={(dense) => onChange({ dense })}
              />
            </div>
            <div className="row">
              <span>
                Kata kunci
                <span className="sub">BM25</span>
              </span>
              <PixelSwitch
                checked={settings.bm25}
                label="Pencarian kata kunci"
                onChange={(bm25) => onChange({ bm25 })}
              />
            </div>
            <div className="readout">
              <span>MODE</span>
              <b>{MODE_LABEL[mode]}</b>
            </div>
            {mode === 'off' && <div className="warn">Pilih minimal satu.</div>}
          </div>
        </div>

        <div className="card green">
          <h3>POTONGAN TEKS</h3>
          <div className="body">
            <div className="chips" role="group" aria-label="Strategi chunking">
              {STRATEGIES.map((strategy) => (
                <PixelChip
                  key={strategy.value}
                  active={settings.strategy === strategy.value}
                  onClick={() => onChange({ strategy: strategy.value })}
                >
                  {strategy.label}
                </PixelChip>
              ))}
            </div>
            <div className="row">
              <span>Jumlah sumber</span>
              <PixelStepper
                value={settings.topK}
                label="Jumlah sumber"
                onChange={(topK) => onChange({ topK })}
              />
            </div>
          </div>
        </div>

        <div className="card dark">
          <h3>DATA</h3>
          <div className="body">
            <div className="kv">
              <span>Koleksi</span>
              <span>{SNAPSHOT.docs} Pokémon · Gen I–III</span>
              <span>Model</span>
              <span>{SNAPSHOT.model}</span>
              <span>Indeks</span>
              <span>{SNAPSHOT.indexVersion}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
