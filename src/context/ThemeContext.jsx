import { createContext, useContext, useEffect, useMemo, useState } from 'react';

const THEME_STORAGE_KEY = 'kg-chat-theme';
const FONT_STORAGE_KEY = 'kg-chat-font';
const COLOR_STORAGE_KEY = 'kg-chat-primary-color';

const FONT_OPTIONS = [
  {
    label: 'Inter',
    value: 'Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
  },
  {
    label: 'Poppins',
    value: 'Poppins, Inter, system-ui, sans-serif'
  },
  {
    label: 'Source Sans 3',
    value: '"Source Sans 3", Inter, system-ui, sans-serif'
  }
];

const COLOR_OPTIONS = [
  { label: 'Blue', value: '#2563eb' },
  { label: 'Emerald', value: '#059669' },
  { label: 'Violet', value: '#7c3aed' }
];

const ThemeContext = createContext(null);

function resolveInitialTheme() {
  let storedTheme = null;
  try {
    storedTheme = localStorage.getItem(THEME_STORAGE_KEY);
  } catch {
    storedTheme = null;
  }

  if (storedTheme === 'dark' || storedTheme === 'light') {
    return storedTheme;
  }

  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  return prefersDark ? 'dark' : 'light';
}

function resolveStoredOption(storageKey, fallbackValue, availableOptions) {
  let stored = null;
  try {
    stored = localStorage.getItem(storageKey);
  } catch {
    stored = null;
  }

  const exists = availableOptions.some((item) => item.value === stored);
  return exists ? stored : fallbackValue;
}

export function ThemeProvider({ children }) {
  const [theme, setTheme] = useState(resolveInitialTheme);
  const [fontFamily, setFontFamily] = useState(() =>
    resolveStoredOption(FONT_STORAGE_KEY, FONT_OPTIONS[0].value, FONT_OPTIONS)
  );
  const [primaryColor, setPrimaryColor] = useState(() =>
    resolveStoredOption(COLOR_STORAGE_KEY, COLOR_OPTIONS[0].value, COLOR_OPTIONS)
  );

  useEffect(() => {
    const root = document.documentElement;
    root.classList.toggle('dark', theme === 'dark');
    root.setAttribute('data-theme', theme);

    try {
      localStorage.setItem(THEME_STORAGE_KEY, theme);
    } catch {
      return undefined;
    }
  }, [theme]);

  useEffect(() => {
    document.documentElement.style.setProperty('--app-font-family', fontFamily);
    try {
      localStorage.setItem(FONT_STORAGE_KEY, fontFamily);
    } catch {
      return undefined;
    }
  }, [fontFamily]);

  useEffect(() => {
    document.documentElement.style.setProperty('--primary-color', primaryColor);
    try {
      localStorage.setItem(COLOR_STORAGE_KEY, primaryColor);
    } catch {
      return undefined;
    }
  }, [primaryColor]);

  const value = useMemo(
    () => ({
      theme,
      setTheme,
      fontFamily,
      setFontFamily,
      primaryColor,
      setPrimaryColor,
      fontOptions: FONT_OPTIONS,
      colorOptions: COLOR_OPTIONS,
      toggleTheme: () => setTheme((current) => (current === 'dark' ? 'light' : 'dark'))
    }),
    [theme, fontFamily, primaryColor]
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useThemePreferences() {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useThemePreferences must be used within ThemeProvider');
  }

  return context;
}