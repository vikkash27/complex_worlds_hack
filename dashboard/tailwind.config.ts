import type { Config } from "tailwindcss";
// @ts-expect-error - no types ship with this util
import flattenColorPalette from "tailwindcss/lib/util/flattenColorPalette";

const config: Config = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      colors: {
        ink: {
          950: "#08090c",
          900: "#0d0f14",
          800: "#10131a",
          700: "#161a23",
          600: "#1e2330",
          500: "#2a2f3d",
        },
      },
      animation: {
        aurora: "aurora 60s linear infinite",
        pulse: "pulse 1.6s infinite",
      },
      keyframes: {
        aurora: {
          from: { backgroundPosition: "50% 50%, 50% 50%" },
          to: { backgroundPosition: "350% 50%, 350% 50%" },
        },
      },
    },
  },
  plugins: [addVariablesForColors],
};

function addVariablesForColors({ addBase, theme }: any) {
  const allColors = flattenColorPalette.default
    ? flattenColorPalette.default(theme("colors"))
    : flattenColorPalette(theme("colors"));
  const newVars = Object.fromEntries(
    Object.entries(allColors).map(([key, val]) => [`--${key}`, val as string])
  );
  addBase({ ":root": newVars });
}

export default config;
