import { useThemePreferences } from '../context/ThemeContext';

function ThemeToggle() {
  const {
    theme,
    toggleTheme,
    fontFamily,
    setFontFamily,
    primaryColor,
    setPrimaryColor,
    fontOptions,
    colorOptions
  } = useThemePreferences();

  return (
    <div className="flex flex-wrap items-center justify-end gap-2">
      <button
        type="button"
        aria-label="Toggle light and dark mode"
        onClick={toggleTheme}
        className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs font-medium text-slate-800 transition hover:bg-slate-100 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100 dark:hover:bg-slate-800"
      >
        {theme === 'dark' ? 'Light Mode' : 'Dark Mode'}
      </button>

      <label className="sr-only" htmlFor="font-select">
        Select font
      </label>
      <select
        id="font-select"
        aria-label="Choose chat font"
        value={fontFamily}
        onChange={(event) => setFontFamily(event.target.value)}
        className="rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-xs text-slate-800 outline-none ring-offset-2 focus:ring-2 focus:ring-[var(--primary-color)] dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
      >
        {fontOptions.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>

      <label className="sr-only" htmlFor="color-select">
        Select primary color
      </label>
      <select
        id="color-select"
        aria-label="Choose primary color"
        value={primaryColor}
        onChange={(event) => setPrimaryColor(event.target.value)}
        className="rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-xs text-slate-800 outline-none ring-offset-2 focus:ring-2 focus:ring-[var(--primary-color)] dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
      >
        {colorOptions.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </div>
  );
}

export default ThemeToggle;