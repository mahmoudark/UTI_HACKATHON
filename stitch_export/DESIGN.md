---
name: Clinical Clarity
colors:
  surface: '#faf8ff'
  surface-dim: '#d2d9f4'
  surface-bright: '#faf8ff'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f2f3ff'
  surface-container: '#eaedff'
  surface-container-high: '#e2e7ff'
  surface-container-highest: '#dae2fd'
  on-surface: '#131b2e'
  on-surface-variant: '#434655'
  inverse-surface: '#283044'
  inverse-on-surface: '#eef0ff'
  outline: '#737686'
  outline-variant: '#c3c6d7'
  surface-tint: '#0053db'
  primary: '#004ac6'
  on-primary: '#ffffff'
  primary-container: '#2563eb'
  on-primary-container: '#eeefff'
  inverse-primary: '#b4c5ff'
  secondary: '#006c49'
  on-secondary: '#ffffff'
  secondary-container: '#6cf8bb'
  on-secondary-container: '#00714d'
  tertiary: '#943700'
  on-tertiary: '#ffffff'
  tertiary-container: '#bc4800'
  on-tertiary-container: '#ffede6'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#dbe1ff'
  primary-fixed-dim: '#b4c5ff'
  on-primary-fixed: '#00174b'
  on-primary-fixed-variant: '#003ea8'
  secondary-fixed: '#6ffbbe'
  secondary-fixed-dim: '#4edea3'
  on-secondary-fixed: '#002113'
  on-secondary-fixed-variant: '#005236'
  tertiary-fixed: '#ffdbcd'
  tertiary-fixed-dim: '#ffb596'
  on-tertiary-fixed: '#360f00'
  on-tertiary-fixed-variant: '#7d2d00'
  background: '#faf8ff'
  on-background: '#131b2e'
  surface-variant: '#dae2fd'
typography:
  headline-xl:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.01em
  headline-md:
    fontFamily: Inter
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
  body-lg:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '400'
    lineHeight: 28px
  body-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-sm:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  label-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.01em
  label-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '600'
    lineHeight: 16px
    letterSpacing: 0.03em
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  base: 8px
  container-max: 1200px
  gutter: 24px
  margin-mobile: 16px
  margin-desktop: 48px
---

## Brand & Style

This design system is built for a premium AI clinical assistant, prioritizing trust, accuracy, and cognitive ease. The personality is "The Quiet Expert"—authoritative but unassuming, designed to fade into the background so the medical data can take center stage.

The style is a refined **Minimalist/SaaS hybrid**. It utilizes expansive whitespace, purposeful typography, and a reduced color palette to lower the user's cognitive load during high-stakes clinical decision-making. High-quality execution is achieved through precise alignment, subtle tonal shifts instead of heavy borders, and a complete absence of decorative or distracting elements.

## Colors

The palette is anchored in a deep navy for maximum legibility and a professional blue for primary actions. The background is a very light, desaturated gray to reduce eye strain compared to pure white.

- **Primary (#2563EB):** Used for critical clinical actions, links, and active states.
- **Success/Verified (#10B981):** Specifically reserved for verified sources, clinical guidelines, and "confirmed" status indicators.
- **Surface/Neutral:** The background uses `#F8FAFC`. Surfaces (cards/modals) use pure `#FFFFFF` to create a subtle layered effect without needing heavy shadows.
- **Text (#0F172A):** Used for all primary body and headline text to ensure WCAG AAA compliance.

## Typography

This design system uses **Inter** for its systematic, utilitarian, and highly legible characteristics. In a medical context, clarity is the highest priority.

- **Hierarchical Scale:** Large headlines are reserved for patient names or primary diagnosis.
- **Line Height:** Tightening the line height slightly for headlines while keeping body text open (1.5x) ensures long clinical notes remain readable.
- **Citations:** Small, semi-bold labels are used for source tags to distinguish them from clinical prose.

## Layout & Spacing

The layout follows a **Fixed Grid** model for the main content area to prevent lines of text from becoming too long for comfortable reading. 

- **Clinical Workspace:** Content is centered with a max-width of 1200px.
- **Sidebar:** A collapsible navigation and history rail sits on the left, using a width of 280px (expanded) or 64px (collapsed).
- **Rhythm:** An 8px linear scale is used. Components are separated by 24px (3 units) to maintain a sense of calm and prevent visual density.

## Elevation & Depth

To maintain a clean SaaS aesthetic, this design system avoids heavy shadows. Depth is communicated through **Tonal Layers** and **Low-Contrast Outlines**.

- **Level 0 (Background):** `#F8FAFC` (The base canvas).
- **Level 1 (Cards/Bubbles):** Pure `#FFFFFF` with a 1px border of `#E2E8F0`. No shadow.
- **Level 2 (Active/Modals):** Pure `#FFFFFF` with a soft, 12% opacity neutral shadow (0px 4px 20px) to indicate interaction or focus.
- **Backdrop:** A light blur (8px) is applied to background elements when a clinical modal is active.

## Shapes

The shape language is "Rounded," conveying a sense of approachability and safety. 

- **Cards & Chat Bubbles:** 0.5rem (8px) corner radius.
- **Input Fields:** 0.5rem (8px).
- **Buttons:** 0.5rem (8px) to maintain consistency with container elements.
- **Status Tags:** Fully rounded (pill) to distinguish them from interactive buttons.

## Components

### Chat Bubbles
AI responses use a white background with a subtle border. User inputs are slightly tinted with a very light blue. Bubbles are not "playful"; they are structured containers for information.

### Source Citations
Sources are presented as small, interactive cards at the bottom of an AI response. They feature a leading icon (e.g., a document or checkmark), a truncated title, and a "Verified" badge in subtle green.

### Buttons
Primary buttons are solid `#2563EB` with white text. Secondary buttons use a white background with a `#E2E8F0` border. No gradients or heavy bevels.

### Inputs
Search and patient data fields use a 1px border that shifts to the primary blue on focus. Placeholder text is a light neutral to keep the UI looking clean.

### Sidebar
The collapsible sidebar uses a light tint of the background color. Icons are 20px, stroke-based (not filled), to maintain a lightweight visual footprint.

### Clinical Cards
Patient summaries or clinical guidelines are housed in cards with a clear header-body-footer structure. Footers often contain "Action" buttons like "Copy to EHR" or "View Source."