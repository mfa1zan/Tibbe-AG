import ThemeToggle from '../components/ThemeToggle';

function SettingsPage() {
  return (
    <section className="app-route-panel">
      <h2 className="app-route-title">Display Settings</h2>
      <p className="app-route-helper">
        Customize theme mode, font family, and primary color for your chat workspace.
      </p>
      <ThemeToggle />
    </section>
  );
}

export default SettingsPage;
