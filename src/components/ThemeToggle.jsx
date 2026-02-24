import { useThemePreferences } from '../context/ThemeContext';
import './ThemeToggle.css';

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
    <div className="theme-toggle-wrap">
      <button
        type="button"
        aria-label="Toggle light and dark mode"
        onClick={toggleTheme}
        className="theme-toggle-button"
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
        className="theme-select"
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
        className="theme-select"
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