"""Draws the two schematic figures (overview and three-step workflow) as SVG
in point units, so that text is printed at 7.5-10 pt when the figure spans
the full text width (516 pt)."""
import textwrap
from pathlib import Path
OUT = Path(__file__).resolve().parents[3] / 'latex'
W = 516
FONT = 'Helvetica, Arial, sans-serif'


def esc(t):
    return t.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


class SVG:
    def __init__(self, w, h):
        self.w, self.h, self.items = w, h, []

    def rect(self, x, y, w, h, fill, stroke, rx=5, sw=1.0, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        self.items.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def text(self, x, y, t, size=7.5, bold=False, color='#212121', anchor='middle', italic=False):
        fw = ' font-weight="bold"' if bold else ''
        fs = ' font-style="italic"' if italic else ''
        self.items.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{color}"{fw}{fs}>{esc(t)}</text>')

    def lines(self, cx, y, lines, size=7.5, lh=None, **kw):
        lh = lh or size * 1.25
        for i, l in enumerate(lines):
            self.text(cx, y + i * lh, l, size, **kw)

    def arrow(self, x1, y1, x2, y2, color='#455a64', sw=1.1):
        self.items.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{sw}" marker-end="url(#a)"/>')

    def raw(self, s):
        self.items.append(s)

    def save(self, name):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}pt" height="{self.h}pt" '
                f'viewBox="0 0 {self.w} {self.h}" font-family="{FONT}">'
                '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">'
                '<path d="M0 0 L10 5 L0 10 z" fill="#455a64"/></marker></defs>')
        (OUT / name).write_text(head + '\n'.join(self.items) + '</svg>', encoding='utf-8')


def box_with_text(s, x, y, w, txt, fill, stroke, size=7.5, bold_first=False, width_chars=None, pad=5):
    width_chars = width_chars or int(w / (size * 0.52))
    paras = txt.split('\n')
    lines = []
    for p in paras:
        lines += textwrap.wrap(p, width_chars) or ['']
    lh = size * 1.25
    h = pad * 2 + lh * len(lines)
    s.rect(x, y, w, h, fill, stroke)
    for i, l in enumerate(lines):
        s.text(x + w / 2, y + pad + size + i * lh - 1, l, size, bold=(bold_first and i == 0))
    return h


# ------------------------------------------------------------------ workflow figure
BANDS = [
    ('Stage 1. Physics-informed model of the CMUT', '#e3f2fd', '#1565c0', '#0d47a1', [
        'Inputs\nfour layer thicknesses: t_e, t_np, t_ox, t_w',
        'Plate model\nbending stiffness D of the Au/Si3N4 layer pair, with the neutral-axis shift',
        'Neural network\nadds a correction g to the plate model for effects it does not include',
        'Training\nfit the 123 FEM designs; plate equation also checked at 600 extra designs']),
    ('Stage 2. Gradient-based design', '#e8f5e9', '#2e7d32', '#1b5e20', [
        'Freeze the model\neach evaluation takes milliseconds',
        'Goal\nhit a target displacement, or make it as large as possible',
        'Gradients\nby automatic differentiation through the model, checked against finite differences',
        'Result\nnominal design θ*']),
    ('Stage 3. Correction for fabrication errors', '#fff3e0', '#e65100', '#bf360c', [
        'Simulated errors\nthickness ±3%, oxide ±5%, wall ±0.08 µm (assumed values)',
        'Sampling\n400 randomly perturbed copies of each design',
        'Objective\naverage error plus its spread, or the average of the worst 5% (CVaR)',
        'Result\nrobust design θ_rob, re-optimized for each target']),
]


def workflow():
    s = SVG(W, 10)
    y = 6
    bw, gap, x0 = 114, 15, 10
    for title, fill, stroke, tcol, boxes in BANDS:
        heights = [box_with_text(SVG(0, 0), 0, 0, bw, b, '#fff', stroke, bold_first=True) for b in boxes]
        bh = max(heights)
        band_h = 22 + bh + 8
        s.rect(4, y, W - 8, band_h, fill, stroke, rx=7, sw=1.2)
        s.text(12, y + 14, title, 9, bold=True, color=tcol, anchor='start')
        for i, b in enumerate(boxes):
            x = x0 + i * (bw + gap)
            box_with_text(s, x, y + 22, bw, b, '#ffffff', stroke, bold_first=True)
            if i < len(boxes) - 1:
                s.arrow(x + bw + 1, y + 22 + bh / 2, x + bw + gap - 1, y + 22 + bh / 2)
        if title.startswith('Stage 3'):
            pass
        else:
            s.arrow(W / 2, y + band_h, W / 2, y + band_h + 10)
        y += band_h + 12
    note = ('Also tested: correction policies learned once and reused for new targets (reinforcement learning and a '
            'gradient-trained network). They did not work on targets outside the training set (Section VI-E).')
    lines = textwrap.wrap(note, 120)
    s.rect(4, y - 4, W - 8, 8 + 10 * len(lines), '#f5f5f5', '#9e9e9e', rx=5, dash='3,2')
    s.lines(W / 2, y + 6, lines, 7.5, lh=10, color='#424242', italic=True)
    s.h = y + 8 + 10 * len(lines)
    s.save('Fig_4.svg')


# ------------------------------------------------------------------ overview figure
def overview():
    s = SVG(W, 10)
    cols = [('1. Physics-informed model', '#e3f2fd', '#1565c0', '#0d47a1',
             'Learns the CMUT response from FEM data, guided by a plate-bending model'),
            ('2. Gradient-based design', '#e8f5e9', '#2e7d32', '#1b5e20',
             'Finds layer thicknesses that meet a target, using gradients through the model'),
            ('3. Robust correction', '#fff3e0', '#e65100', '#bf360c',
             'Adjusts the design so that more devices stay within specification despite fabrication errors')]
    bw, gap = 154, 17
    hs = []
    for i, (t, f, st, tc, body) in enumerate(cols):
        x = 8 + i * (bw + gap)
        lines = textwrap.wrap(body, 33)
        h = 24 + 11 * len(lines) + 6
        hs.append(h)
        s.rect(x, 6, bw, h, f, st, rx=7, sw=1.2)
        s.text(x + bw / 2, 20, t, 9, bold=True, color=tc)
        s.lines(x + bw / 2, 34, lines, 7.5, lh=11)
        if i < 2:
            s.arrow(x + bw + 2, 6 + h / 2, x + bw + gap - 2, 6 + h / 2)
    y = 6 + max(hs) + 12
    # testbench panels
    ph = 118
    s.rect(8, y, 250, ph, '#f3e5f5', '#6a1b9a', rx=7, sw=1.2)
    s.text(133, y + 14, 'Test case 1: CMUT (full workflow)', 9, bold=True, color='#4a148c')
    s.text(133, y + 26, '123 FEM designs; four layer thicknesses', 7.5, color='#424242')
    # simple CMUT cross-section
    by = y + 40
    s.rect(28, by + 52, 210, 10, '#90a4ae', '#546e7a', rx=1)      # substrate
    s.rect(28, by + 44, 210, 8, '#ffcc80', '#ef6c00', rx=1)       # oxide
    s.rect(40, by + 14, 14, 30, '#b0bec5', '#546e7a', rx=1)       # wall
    s.rect(212, by + 14, 14, 30, '#b0bec5', '#546e7a', rx=1)
    s.raw(f'<path d="M 54 {by+14} Q 133 {by+34} 212 {by+14}" fill="none" stroke="#e53935" stroke-width="2"/>')
    s.raw(f'<path d="M 54 {by+9} Q 133 {by+29} 212 {by+9}" fill="none" stroke="#1565c0" stroke-width="3"/>')
    s.text(133, by + 4, 'membrane: Si3N4 (t_np) on Au electrode (t_e)', 7.5, color='#1565c0')
    s.text(133, by + 42, 'vacuum gap', 7.5, color='#607d8b')
    s.text(133, by + 74, 'walls (t_w) and oxide (t_ox) on the substrate', 7.5, color='#424242')
    s.rect(266, y, 242, ph, '#e0f7fa', '#00838f', rx=7, sw=1.2)
    s.text(387, y + 14, 'Test case 2: photonic waveguide bend', 9, bold=True, color='#006064')
    s.text(387, y + 26, 'stages 2 and 3 with exact gradients', 7.5, color='#424242')
    gx, gy = 300, y + 40
    s.rect(gx, gy, 170, 62, '#ffffff', '#00838f', rx=2)
    s.rect(gx, gy + 40, 60, 10, '#4dd0e1', '#00838f', rx=0)
    s.rect(gx + 110, gy, 10, 25, '#4dd0e1', '#00838f', rx=0)
    s.raw(f'<path d="M {gx+60} {gy+45} Q {gx+115} {gy+45} {gx+115} {gy+25}" fill="none" stroke="#4dd0e1" stroke-width="10"/>')
    s.text(gx + 85, gy + 74, 'light enters left, leaves at the top', 7.5, color='#424242')
    y += ph + 10
    msg = 'Physics-informed, gradient-based, fabrication-aware inverse design'
    s.rect(8, y, W - 16, 22, '#eceff1', '#546e7a', rx=6)
    s.text(W / 2, y + 15, msg, 9, bold=True, color='#263238')
    s.h = y + 26
    s.save('abstract_Fig.svg')


if __name__ == '__main__':
    workflow(); overview(); print('diagrams written')
