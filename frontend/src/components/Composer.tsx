import { useState, type FormEvent } from 'react';

interface ComposerProps {
  disabled: boolean;
  onSend: (text: string) => void;
}

export function Composer({ disabled, onSend }: ComposerProps) {
  const [value, setValue] = useState('');

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const text = value.trim();
    if (!text || disabled) return;
    onSend(text);
    setValue('');
  };

  return (
    <div className="composer-wrap">
      <form className="composer" onSubmit={submit}>
        <label className="prompt" htmlFor="composer-input">
          Mau tanya
          <br />
          apa?
        </label>
        <input
          id="composer-input"
          value={value}
          onChange={(event) => setValue(event.target.value)}
          placeholder="Tanya tentang Pokémon…"
          aria-label="Pertanyaan"
          autoComplete="off"
          disabled={disabled}
        />
        <button type="submit" className="btn red" disabled={disabled || !value.trim()}>
          TANYA
        </button>
      </form>
      {disabled && <p className="composer-warn">Pilih minimal satu sumber di Setelan.</p>}
    </div>
  );
}
