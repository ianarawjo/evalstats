/* dark.js — shared dark-mode toggle for all pages
   Expects: <button id="dark-toggle"> and <svg id="dark-icon"> in the nav.
   Reads/writes localStorage key "theme" = "dark" | "light".
   Also respects prefers-color-scheme when no saved preference. */
(function () {
  const root = document.documentElement;
  const btn  = document.getElementById('dark-toggle');
  const icon = document.getElementById('dark-icon');
  if (!btn || !icon) return;

  const SUN  = '<path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/><circle cx="12" cy="12" r="5"/>';
  const MOON = '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>';

  function setDark(dark) {
    if (dark) {
      root.setAttribute('data-theme', 'dark');
      icon.innerHTML = SUN;
    } else {
      root.removeAttribute('data-theme');
      icon.innerHTML = MOON;
    }
    try { localStorage.setItem('theme', dark ? 'dark' : 'light'); } catch (e) {}
  }

  const saved      = (() => { try { return localStorage.getItem('theme'); } catch (e) { return null; } })();
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  setDark(saved === 'dark' || (!saved && prefersDark));

  btn.addEventListener('click', () => {
    setDark(root.getAttribute('data-theme') !== 'dark');
  });
})();
