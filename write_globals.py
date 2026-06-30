from pathlib import Path
p = Path('frontend/app/globals.css')
content = '''@import "tailwindcss";

/* Core design tokens */
:root {
  --background: #ffffff;
  --foreground: #171717;
}

@media (prefers-color-scheme: dark) {
  :root {
    --background: #0a0a0a;
    --foreground: #ededed;
  }
}

/* Global resets and sensible defaults */
*, *::before, *::after {
  box-sizing: border-box;
}

html {
  font-size: 16px;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  -webkit-text-size-adjust: 100%;
}

body {
  margin: 0;
  background: var(--background);
  color: var(--foreground);
  font-family: Arial, Helvetica, sans-serif;
  line-height: 1.6;
}

/* Optional: reduce visual noise from oversized scrollbars */
* {
  scrollbar-width: thin;
  scrollbar-color: rgba(0,0,0,0.15) transparent;
}

/* KaTeX / math counters (ensure no duplicate counter resets) */
html { counter-reset: katexEqnNo mmlEqnNo; }

/* Provide a safe @font-face placeholder if a variable-font file is shipped
   The actual font file may be emitted into the build output with a hashed name.
   If you have a specific woff2 in `public/media/`, add or adjust this rule. */
/*
@font-face {
  font-family: 'Source Sans VF';
  font-weight: 100 900;
  font-style: normal;
  src: url('/media/SourceSansVF-Upright.woff2') format('woff2');
  font-display: swap;
}
*/

/* Utility: reduce tap highlight on mobile */
:root { --webkit-tap-highlight-color: rgba(0,0,0,0); }
'''
p.write_text(content, encoding='utf-8')
print('written', p)
