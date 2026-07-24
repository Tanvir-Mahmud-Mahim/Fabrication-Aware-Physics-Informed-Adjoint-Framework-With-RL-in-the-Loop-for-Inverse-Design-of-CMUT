const pptxgen = require("pptxgenjs");
const p = new pptxgen();
p.layout = "LAYOUT_WIDE"; // 13.3 x 7.5
p.author = "Tanvir M. Mahim";
p.title = "PARL-ID Figures (editable)";
const F = "Arial";
const NAVY="0D47A1", BLUE="1565C0", LBLUE="E3F2FD", MBLUE="BBDEFB",
      GREEN="2E7D32", DGREEN="1B5E20", LGREEN="E8F5E9", MGREEN="C8E6C9",
      ORANGE="E65100", DORANGE="BF360C", LORANGE="FFF3E0", MORANGE="FFE0B2",
      PURPLE="6A1B9A", DPURPLE="4A148C", LPURPLE="F3E5F5",
      GRAY="37474F", INK="263238";

function card(s, x, y, w, h, fill, line, opts={}) {
  s.addShape(p.shapes.ROUNDED_RECTANGLE, Object.assign(
    { x, y, w, h, fill: { color: fill }, line: { color: line, width: 1.2 }, rectRadius: 0.08 }, opts));
}
function box(s, x, y, w, h, fill, line) {
  s.addShape(p.shapes.ROUNDED_RECTANGLE,
    { x, y, w, h, fill: { color: fill }, line: { color: line, width: 0.8 }, rectRadius: 0.05 });
}
function arrow(s, x, y, w, h, color=GRAY, dash) {
  // PowerPoint rejects negative extents: normalize to positive w/h and flip.
  let flipH = false, flipV = false;
  if (w < 0) { x += w; w = -w; flipH = true; }
  if (h < 0) { y += h; h = -h; flipV = true; }
  s.addShape(p.shapes.LINE, { x, y, w, h, flipH, flipV,
    line: { color, width: 1.6, endArrowType: "triangle", dashType: dash || "solid" } });
}
function txt(s, t, x, y, w, h, size, color, o={}) {
  s.addText(t, Object.assign({ x, y, w, h, fontSize: size, color, fontFace: F,
    align: "center", valign: "middle", margin: 0 }, o));
}

/* ---------------- S1: title ---------------- */
let s = p.addSlide();
s.background = { color: "1E2761" };
txt(s, "PARL-ID", 0.8, 1.6, 11.7, 1.0, 54, "FFFFFF", { bold: true, align: "left" });
txt(s, "A Fabrication-Aware Physics-Informed Adjoint Framework With Reinforcement\nLearning in the Loop for Inverse Design of CMUT and Photonic Sensors",
    0.8, 2.7, 11.7, 1.2, 22, "CADCFC", { align: "left" });
txt(s, "Editable figure pack — IEEE Sensors Journal manuscript", 0.8, 4.3, 11.7, 0.5, 16, "90A4AE", { align: "left", italic: true });
txt(s, "T. M. Mahim  ·  M. M. Rahman  ·  A.H.M.A. Rahim   |   BRAC University", 0.8, 5.0, 11.7, 0.5, 14, "CADCFC", { align: "left" });

/* ---------------- S2: graphical abstract ---------------- */
s = p.addSlide();
s.background = { color: "FFFFFF" };
txt(s, "Graphical Abstract", 0.5, 0.18, 12.3, 0.5, 24, INK, { bold: true, align: "left" });

card(s, 0.5, 0.85, 3.9, 3.1, LBLUE, BLUE);
txt(s, "1. Multi-Physics PINN", 0.6, 0.95, 3.7, 0.4, 16, NAVY, { bold: true });
txt(s, "forward surrogate", 0.6, 1.3, 3.7, 0.3, 11, NAVY);
box(s, 0.75, 1.7, 1.65, 0.6, "FFFFFF", BLUE);
txt(s, "FEM / FDFD database\n(data loss)", 0.75, 1.7, 1.65, 0.6, 9, INK);
box(s, 2.55, 1.7, 1.65, 0.6, "FFFFFF", BLUE);
txt(s, "Governing PDEs\n(residual loss)", 2.55, 1.7, 1.65, 0.6, 9, INK);
arrow(s, 1.6, 2.32, 0.6, 0.33);
arrow(s, 3.35, 2.32, -0.6, 0.33);
box(s, 1.2, 2.7, 2.5, 0.6, MBLUE, BLUE);
txt(s, "Fourier-feature MLP\nhard-BC encoding", 1.2, 2.7, 2.5, 0.6, 10, INK, { bold: true });
txt(s, "Plate electro-mechanics + acoustics (CMUT)\nHelmholtz equation (photonic)", 0.6, 3.4, 3.7, 0.5, 8.5, GRAY);

card(s, 4.7, 0.85, 3.9, 3.1, LGREEN, GREEN);
txt(s, "2. Neural-Adjoint", 4.8, 0.95, 3.7, 0.4, 16, DGREEN, { bold: true });
txt(s, "inverse engine", 4.8, 1.3, 3.7, 0.3, 11, DGREEN);
box(s, 4.95, 1.7, 3.4, 0.6, "FFFFFF", GREEN);
txt(s, "frozen surrogate, objective J(θ)\nautodiff  =  classical adjoint", 4.95, 1.7, 3.4, 0.6, 9.5, INK);
arrow(s, 6.65, 2.32, 0, 0.15);
box(s, 4.95, 2.5, 3.4, 0.6, MGREEN, GREEN);
txt(s, "projected gradient descent\nfabrication-feasible set Π", 4.95, 2.5, 3.4, 0.6, 9.5, INK);
arrow(s, 6.65, 3.12, 0, 0.15);
box(s, 5.4, 3.3, 2.5, 0.45, "FFFFFF", DGREEN);
txt(s, "nominal design θ*", 5.4, 3.3, 2.5, 0.45, 11, INK, { bold: true });

card(s, 8.9, 0.85, 3.9, 3.1, LORANGE, ORANGE);
txt(s, "3. RL Fabrication Loop", 9.0, 0.95, 3.7, 0.4, 16, DORANGE, { bold: true });
txt(s, "yield-aware correction policy", 9.0, 1.3, 3.7, 0.3, 11, DORANGE);
box(s, 9.05, 1.7, 1.7, 0.75, "FFFFFF", ORANGE);
txt(s, "GCN-SAC agent\ndevice-graph state\nprioritized replay", 9.05, 1.7, 1.7, 0.75, 8.5, INK);
box(s, 10.95, 1.7, 1.7, 0.75, MORANGE, ORANGE);
txt(s, "virtual fab Ξ\nthickness ±3%\netch ±80 nm, stress", 10.95, 1.7, 1.7, 0.75, 8.5, INK);
arrow(s, 10.75, 1.95, 0.2, 0);
arrow(s, 10.95, 2.25, -0.2, 0);
box(s, 9.05, 2.6, 3.6, 0.5, "FFFFFF", ORANGE);
txt(s, "tail-risk reward = CVaR₅% over K process corruptions", 9.05, 2.6, 3.6, 0.5, 8.5, INK);
arrow(s, 10.85, 3.12, 0, 0.15);
box(s, 9.55, 3.3, 2.6, 0.45, "FFFFFF", DORANGE);
txt(s, "robust design + policy", 9.55, 3.3, 2.6, 0.45, 11, INK, { bold: true });

arrow(s, 4.4, 2.4, 0.3, 0); arrow(s, 8.6, 2.4, 0.3, 0);

card(s, 0.5, 4.25, 5.9, 2.2, LPURPLE, PURPLE);
txt(s, "Testbench I: Unit-cell CMUT", 0.6, 4.35, 5.7, 0.35, 13, DPURPLE, { bold: true });
txt(s, "open FEM database (released): 369 combos × 150 freq. points", 0.6, 4.7, 5.7, 0.3, 9, GRAY);
s.addShape(p.shapes.RECTANGLE, { x: 1.4, y: 5.9, w: 4.1, h: 0.22, fill: { color: "90A4AE" }, line: { color: "455A64", width: 0.8 } });
s.addShape(p.shapes.RECTANGLE, { x: 1.7, y: 5.72, w: 3.5, h: 0.18, fill: { color: "FFCC80" }, line: { color: ORANGE, width: 0.8 } });
s.addShape(p.shapes.RECTANGLE, { x: 1.7, y: 5.35, w: 0.28, h: 0.37, fill: { color: "B0BEC5" }, line: { color: "455A64", width: 0.8 } });
s.addShape(p.shapes.RECTANGLE, { x: 4.92, y: 5.35, w: 0.28, h: 0.37, fill: { color: "B0BEC5" }, line: { color: "455A64", width: 0.8 } });
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 1.98, y: 5.5, w: 2.94, h: 0.16, fill: { color: MBLUE }, line: { color: BLUE, width: 1.4 }, rectRadius: 0.08, rotate: 358 });
txt(s, "membrane deflection → frequency–displacement profile", 0.6, 6.15, 5.7, 0.3, 9, GRAY);

card(s, 6.9, 4.25, 5.9, 2.2, "E0F7FA", "00838F");
txt(s, "Testbench II: Photonic benchmarks", 7.0, 4.35, 5.7, 0.35, 13, "006064", { bold: true });
txt(s, "ceviche-challenges / invrs-gym — exact adjoints, open leaderboards", 7.0, 4.7, 5.7, 0.3, 9, GRAY);
s.addShape(p.shapes.RECTANGLE, { x: 7.6, y: 5.15, w: 4.4, h: 1.0, fill: { color: "FFFFFF" }, line: { color: "00838F", width: 0.8 } });
s.addShape(p.shapes.RECTANGLE, { x: 7.6, y: 5.52, w: 1.3, h: 0.26, fill: { color: "4DD0E1" } });
s.addShape(p.shapes.RECTANGLE, { x: 10.7, y: 5.28, w: 1.3, h: 0.2, fill: { color: "4DD0E1" } });
s.addShape(p.shapes.RECTANGLE, { x: 10.7, y: 5.82, w: 1.3, h: 0.2, fill: { color: "4DD0E1" } });
s.addShape(p.shapes.LINE, { x: 8.9, y: 5.38, w: 1.8, h: 0.27, flipV: true, line: { color: "4DD0E1", width: 8 } });
s.addShape(p.shapes.LINE, { x: 8.9, y: 5.65, w: 1.8, h: 0.27, line: { color: "4DD0E1", width: 8 } });
txt(s, "mode converter / WDM as sensor building blocks", 7.0, 6.15, 5.7, 0.3, 9, GRAY);

txt(s, "PARL-ID: physics-informed, gradient-exact, fabrication-aware inverse design — one architecture, two sensor domains",
    0.5, 6.75, 12.3, 0.45, 12, INK, { bold: true, fill: { color: "ECEFF1" } });

/* ---------------- S3: CMUT device ---------------- */
s = p.addSlide();
s.background = { color: "FFFFFF" };
txt(s, "Fig. 1 — Unit-Cell CNT-Implanted Piston-Shaped CMUT", 0.5, 0.18, 12.3, 0.5, 22, INK, { bold: true, align: "left" });

txt(s, "a) Top view", 0.7, 0.85, 3.6, 0.35, 14, INK, { bold: true });
s.addShape(p.shapes.RECTANGLE, { x: 0.9, y: 1.3, w: 3.4, h: 3.4, fill: { color: "ECEFF1" }, line: { color: "455A64", width: 1.2 } });
s.addShape(p.shapes.RECTANGLE, { x: 1.75, y: 2.15, w: 1.7, h: 1.7, fill: { color: "FFD54F" }, line: { color: "F57F17", width: 1 } });
for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++)
  s.addShape(p.shapes.OVAL, { x: 1.93 + i * 0.29, y: 2.33 + j * 0.29, w: 0.1, h: 0.1, fill: { color: "5D4037" } });
s.addShape(p.shapes.RECTANGLE, { x: 2.3, y: 1.3, w: 0.6, h: 0.3, fill: { color: "B0BEC5" }, line: { color: "455A64", width: 0.8 } });
txt(s, "via", 2.3, 1.3, 0.6, 0.3, 8, INK);
txt(s, "l = 63.5 µm", 0.9, 4.75, 3.4, 0.3, 10, INK);
txt(s, "l_et = 33 µm (top electrode, Au) — CNT array at centre", 0.7, 5.1, 3.8, 0.5, 9, GRAY);

txt(s, "b) Side view (cross-section)", 4.9, 0.85, 4.4, 0.35, 14, INK, { bold: true });
const bx = 5.0, by = 1.35;
s.addShape(p.shapes.RECTANGLE, { x: bx, y: by + 2.4, w: 4.4, h: 0.55, fill: { color: "78909C" }, line: { color: GRAY, width: 1 } });
txt(s, "Si substrate", bx, by + 2.4, 4.4, 0.55, 10, "FFFFFF");
s.addShape(p.shapes.RECTANGLE, { x: bx + 0.25, y: by + 2.1, w: 3.9, h: 0.3, fill: { color: "FFD54F" }, line: { color: "F57F17", width: 0.8 } });
txt(s, "bottom electrode (Au), l_eb = 35 µm", bx + 0.25, by + 2.1, 3.9, 0.3, 8, "5D4037");
s.addShape(p.shapes.RECTANGLE, { x: bx, y: by + 1.0, w: 0.65, h: 1.1, fill: { color: "FFAB91" }, line: { color: "D84315", width: 0.8 } });
s.addShape(p.shapes.RECTANGLE, { x: bx + 3.75, y: by + 1.0, w: 0.65, h: 1.1, fill: { color: "FFAB91" }, line: { color: "D84315", width: 0.8 } });
txt(s, "SiO₂", bx, by + 1.4, 0.65, 0.3, 9, "BF360C");
txt(s, "SiO₂", bx + 3.75, by + 1.4, 0.65, 0.3, 9, "BF360C");
s.addShape(p.shapes.RECTANGLE, { x: bx + 0.65, y: by + 1.0, w: 3.1, h: 1.1, fill: { color: "FFFFFF" }, line: { color: "90A4AE", width: 0.6, dashType: "dash" } });
txt(s, "vacuum gap g", bx + 0.65, by + 1.35, 3.1, 0.4, 10, "607D8B");
s.addShape(p.shapes.RECTANGLE, { x: bx + 0.35, y: by + 0.7, w: 3.7, h: 0.3, fill: { color: "FFD54F" }, line: { color: "F57F17", width: 0.8 } });
s.addShape(p.shapes.RECTANGLE, { x: bx, y: by + 0.25, w: 4.4, h: 0.45, fill: { color: "81D4FA" }, line: { color: "0277BD", width: 0.8 } });
txt(s, "Si₃N₄ passivation", bx, by + 0.25, 4.4, 0.45, 10, "01579B");
for (let i = 0; i < 5; i++)
  s.addShape(p.shapes.RECTANGLE, { x: bx + 0.9 + i * 0.65, y: by + 1.0, w: 0.06, h: 0.28, fill: { color: "3E2723" } });
txt(s, "t_np", bx - 0.5, by + 0.25, 0.45, 0.45, 10, "0277BD", { bold: true, align: "right" });
txt(s, "t_e", bx - 0.5, by + 0.7, 0.45, 0.3, 10, ORANGE, { bold: true, align: "right" });
txt(s, "t_ox →", bx + 4.45, by + 1.35, 0.7, 0.4, 10, "BF360C", { bold: true, align: "left" });
txt(s, "t_w (wall)", bx - 0.05, by + 3.0, 1.2, 0.3, 10, "BF360C", { bold: true });
txt(s, "CNTs: r = 0.15 µm, h = 1.4 µm  |  V_a = 5 V, p_max = 1 MPa, R_load = 1 GΩ", 4.9, 5.35, 4.7, 0.5, 9, GRAY);

txt(s, "c) Three-dimensional view", 10.0, 0.85, 3.0, 0.35, 14, INK, { bold: true });
s.addShape(p.shapes.RECTANGLE, { x: 10.1, y: 3.6, w: 2.7, h: 0.55, fill: { color: "78909C" }, line: { color: GRAY, width: 1 }, rotate: 351 });
s.addShape(p.shapes.RECTANGLE, { x: 10.2, y: 2.9, w: 2.5, h: 0.45, fill: { color: "FFAB91" }, line: { color: "D84315", width: 1 }, rotate: 351 });
s.addShape(p.shapes.RECTANGLE, { x: 10.3, y: 2.25, w: 2.3, h: 0.4, fill: { color: "81D4FA" }, line: { color: "0277BD", width: 1 }, rotate: 351 });
s.addShape(p.shapes.RECTANGLE, { x: 10.75, y: 2.18, w: 1.3, h: 0.22, fill: { color: "FFD54F" }, line: { color: "F57F17", width: 0.8 }, rotate: 351 });
for (let i = 0; i < 4; i++)
  s.addShape(p.shapes.RECTANGLE, { x: 10.9 + i * 0.28, y: 1.98 - i * 0.03, w: 0.05, h: 0.2, fill: { color: "3E2723" } });
arrow(s, 11.3, 1.45, 0, 0.35, "C62828"); arrow(s, 11.9, 1.4, 0, 0.35, "C62828");
txt(s, "ultrasonic excitation", 10.5, 1.05, 2.2, 0.3, 9, "C62828");
txt(s, "piston-shaped composite membrane\nover vacuum cavity", 9.9, 4.5, 3.2, 0.6, 9, GRAY);

/* ---------------- S4: framework ---------------- */
s = p.addSlide();
s.background = { color: "FFFFFF" };
txt(s, "Fig. 4 — PARL-ID Framework (hierarchical execution)", 0.5, 0.18, 12.3, 0.5, 22, INK, { bold: true, align: "left" });

card(s, 0.4, 0.85, 4.0, 5.0, LBLUE, BLUE);
txt(s, "Stage 1 — Multi-Physics PINN Surrogate", 0.5, 0.95, 3.8, 0.4, 13, NAVY, { bold: true });
box(s, 0.6, 1.45, 1.75, 0.75, "FFFFFF", BLUE);
txt(s, "inputs: (x, y),\ndesign θ, frequency ω", 0.6, 1.45, 1.75, 0.75, 8.5, INK);
box(s, 2.45, 1.45, 1.75, 0.75, "FFFFFF", BLUE);
txt(s, "supervision: CMUT FEM\ndatabase / FDFD fields", 2.45, 1.45, 1.75, 0.75, 8.5, INK);
arrow(s, 1.5, 2.22, 0.6, 0.28); arrow(s, 3.3, 2.22, -0.6, 0.28);
box(s, 1.0, 2.55, 2.8, 0.5, MBLUE, BLUE);
txt(s, "Random Fourier features γ(x)", 1.0, 2.55, 2.8, 0.5, 10, INK, { bold: true });
arrow(s, 2.4, 3.07, 0, 0.15);
box(s, 1.0, 3.25, 2.8, 0.5, MBLUE, BLUE);
txt(s, "tanh MLP + hard-BC encoding φ²·N", 1.0, 3.25, 2.8, 0.5, 10, INK, { bold: true });
arrow(s, 2.4, 3.77, 0, 0.15);
box(s, 0.6, 3.95, 1.75, 0.7, "FFFFFF", BLUE);
txt(s, "CMUT head: (W, M)\nplate + acoustics", 0.6, 3.95, 1.75, 0.7, 8.5, INK);
box(s, 2.45, 3.95, 1.75, 0.7, "FFFFFF", BLUE);
txt(s, "photonic head: (Re E, Im E)\nHelmholtz residual", 2.45, 3.95, 1.75, 0.7, 8.5, INK);
box(s, 0.6, 4.8, 3.6, 0.85, "C5CAE9", "283593");
txt(s, "Composite loss (adaptive weights)\nλ_d data MSE + λ_p PDE residuals + λ_b BC", 0.6, 4.8, 3.6, 0.85, 9, "1A237E", { bold: true });

card(s, 4.65, 0.85, 4.0, 5.0, LGREEN, GREEN);
txt(s, "Stage 2 — Neural-Adjoint Inverse Engine", 4.75, 0.95, 3.8, 0.4, 13, DGREEN, { bold: true });
box(s, 4.85, 1.45, 3.6, 0.7, "FFFFFF", GREEN);
txt(s, "freeze trained surrogate N(·; θ)\ntarget spec → objective J(θ)", 4.85, 1.45, 3.6, 0.7, 9, INK);
arrow(s, 6.65, 2.17, 0, 0.15);
box(s, 4.85, 2.35, 3.6, 0.85, MGREEN, GREEN);
txt(s, "reverse-mode autodiff ≡ classical adjoint\n(∂R/∂u)ᵀλ = −(∂J/∂u)ᵀ,  dJ/dθ = J_θ + λᵀR_θ\nvalidated vs. exact FDFD discrete adjoint", 4.85, 2.35, 3.6, 0.85, 8.5, INK);
arrow(s, 6.65, 3.22, 0, 0.15);
box(s, 4.85, 3.4, 3.6, 0.8, "FFFFFF", GREEN);
txt(s, "projected gradient descent, multi-restart\nθ ← Π_F [θ − η∇J]   (bounds, min-feature,\nerosion–dilation for photonic densities)", 4.85, 3.4, 3.6, 0.8, 8.5, INK);
arrow(s, 6.65, 4.22, 0, 0.15);
box(s, 5.5, 4.4, 2.3, 0.5, "A5D6A7", DGREEN);
txt(s, "nominal optimum θ*", 5.5, 4.4, 2.3, 0.5, 11, INK, { bold: true });
txt(s, "ms-level surrogate evaluations make the loop fast enough to host RL (Stage 3)",
    4.85, 5.05, 3.6, 0.6, 8.5, "33691E", { italic: true });

card(s, 8.9, 0.85, 4.0, 5.0, LORANGE, ORANGE);
txt(s, "Stage 3 — RL Virtual Fabrication Loop", 9.0, 0.95, 3.8, 0.4, 13, DORANGE, { bold: true });
box(s, 9.1, 1.45, 1.75, 0.9, "FFFFFF", ORANGE);
txt(s, "GCN-SAC π(a|s)\ndevice graph + PER\nevo warm start", 9.1, 1.45, 1.75, 0.9, 8.5, INK);
box(s, 11.0, 1.45, 1.75, 0.9, MORANGE, ORANGE);
txt(s, "virtual fab Ξ(θ)\nthickness ~N(1,3%)\netch ~N(0,80 nm)\nstress ~N(1,5%)", 11.0, 1.45, 1.75, 0.9, 8, INK);
arrow(s, 10.85, 1.7, 0.15, 0); arrow(s, 11.0, 2.1, -0.15, 0);
box(s, 9.1, 2.55, 3.65, 0.6, "FFFFFF", ORANGE);
txt(s, "K corrupted designs evaluated through\nfrozen PINN surrogate (ms per evaluation)", 9.1, 2.55, 3.65, 0.6, 8.5, INK);
arrow(s, 10.9, 3.17, 0, 0.15);
box(s, 9.1, 3.35, 3.65, 0.55, MORANGE, ORANGE);
txt(s, "tail-risk reward  r = CVaR₅% of performance", 9.1, 3.35, 3.65, 0.55, 10, INK, { bold: true });
arrow(s, 10.9, 3.92, 0, 0.15);
box(s, 9.3, 4.1, 3.25, 0.5, "FFFFFF", DORANGE);
txt(s, "robust design θ_rob + correction policy", 9.3, 4.1, 3.25, 0.5, 10, INK, { bold: true });
txt(s, "policy conditions on (θ, θ*) ⇒ zero-shot transfer\nto new target specifications", 9.1, 4.75, 3.65, 0.6, 8.5, DORANGE, { italic: true });

arrow(s, 4.4, 3.3, 0.25, 0); arrow(s, 8.65, 3.3, 0.25, 0);

txt(s, "Execution: database + PDE physics → train PINN → freeze → adjoint inverse design → θ* → GCN-SAC in virtual fab → θ_rob → FEM/FDFD verification",
    0.4, 6.15, 12.5, 0.5, 11, DPURPLE, { bold: true, fill: { color: LPURPLE } });

/* ---------------- result figure slides ---------------- */
const figs = [
  ["Fig_pinn_train", "Surrogate training on the released FEM database", "(a) convergence across database fractions (10/50/100%); (b) held-out combination parity, R² = 0.86"],
  ["Fig_field", "Forward-model reconstruction", "(a) clamped CMUT membrane deflection; (b) FDFD waveguide field E_z (photonic testbench)"],
  ["Fig_grad", "Adjoint gradient validation", "(a) exact discrete adjoint map; (b) adjoint vs. finite differences, cosine similarity ≈ 1 − 10⁻¹²"],
  ["Fig_inv", "Inverse design at f_tgt = 4.3 MHz", "(a) probe-seeded gradient engine: J* = 0.0813 µm in 360 queries — random search needs 4,817 (13.4×); (b) landscape slice with θ*"],
  ["Fig_yield", "Virtual fabrication loop — yield analysis", "(a) CDFs: CVaR-corrected design dominates on every FOM — mean +5.5%, P5 +5.2%, CVaR5 +5.0%; (b) yield-reward improvement"],
  ["Fig_transfer", "Correction transfer across target specifications", "static corrections destroy value on average (−65% recovery) — motivating the specification-conditioned RL policy"],
  ["Fig_ablation", "Component ablation — every choice individually justified", "(a) inverse engine at equal 360-query budget: composition +11% over best single component; (b) CVaR dominates mean-variance on every yield FOM (+5.0–5.5%)"],
];
for (const [name, title, cap] of figs) {
  s = p.addSlide();
  s.background = { color: "FFFFFF" };
  txt(s, title, 0.5, 0.22, 12.3, 0.5, 22, INK, { bold: true, align: "left" });
  s.addImage({ path: `/tmp/figs/${name}.png`, x: 1.4, y: 1.05, w: 10.5, h: 5.0, sizing: { type: "contain", w: 10.5, h: 5.0 } });
  txt(s, cap, 0.5, 6.4, 12.3, 0.6, 12, GRAY, { italic: true });
  txt(s, `latex/${name}.svg  ·  regenerated by code/experiments/make_figures/`, 0.5, 7.0, 12.3, 0.3, 9, "90A4AE");
}


/* ---------------- FOM summary table ---------------- */
s = p.addSlide();
s.background = { color: "FFFFFF" };
txt(s, "Quantitative FOMs vs. State of the Art (released CMUT database)", 0.5, 0.22, 12.3, 0.5, 22, INK, { bold: true, align: "left" });
const hdr = { fill: { color: "1E2761" }, color: "FFFFFF", bold: true, fontSize: 10 };
const sec = { fill: { color: "ECEFF1" }, color: "263238", bold: true, fontSize: 10 };
const our = { fill: { color: "E8F5E9" }, bold: true, fontSize: 10 };
const cell = { fontSize: 10 };
s.addTable([
 [{ text: "(A) Surrogate accuracy", options: sec }, { text: "Split", options: sec }, { text: "R²", options: sec }, { text: "MAE", options: sec }],
 [{ text: "Linear Regression [prev]", options: cell }, { text: "row†", options: cell }, { text: "0.50", options: cell }, { text: "0.150†", options: cell }],
 [{ text: "Random Forest [prev]", options: cell }, { text: "row†", options: cell }, { text: "0.70", options: cell }, { text: "2.8e-2†", options: cell }],
 [{ text: "Decision Tree [prev]", options: cell }, { text: "row†", options: cell }, { text: "0.71", options: cell }, { text: "1.3e-2†", options: cell }],
 [{ text: "GRU-Attention [prev]", options: cell }, { text: "row†", options: cell }, { text: "0.92", options: cell }, { text: "1.5e-4†", options: cell }],
 [{ text: "This work", options: our }, { text: "row", options: our }, { text: "0.95", options: our }, { text: "2.3e-3 µm", options: our }],
 [{ text: "This work", options: our }, { text: "held-out combo", options: our }, { text: "0.86", options: our }, { text: "3.9e-3 µm", options: our }],
], { x: 0.5, y: 0.95, w: 6.0, colW: [2.4, 1.4, 0.9, 1.3], border: { pt: 0.5, color: "B0BEC5" } });
s.addTable([
 [{ text: "(B) Inverse engine (4.3 MHz)", options: sec }, { text: "Best J [µm]", options: sec }, { text: "Queries", options: sec }, { text: "To match J*", options: sec }],
 [{ text: "Random search", options: cell }, { text: "0.0705", options: cell }, { text: "360", options: cell }, { text: "4,817", options: cell }],
 [{ text: "Unseeded gradient", options: cell }, { text: "0.0731", options: cell }, { text: "400", options: cell }, { text: "stalls", options: cell }],
 [{ text: "PARL-ID seeded gradient", options: our }, { text: "0.0813", options: our }, { text: "360", options: our }, { text: "360  (13.4×)", options: our }],
], { x: 6.9, y: 0.95, w: 5.9, colW: [2.3, 1.2, 1.0, 1.4], border: { pt: 0.5, color: "B0BEC5" } });
s.addTable([
 [{ text: "(C) Robustness (2×10³ common MC draws)", options: sec }, { text: "µ [µm]", options: sec }, { text: "σ/µ", options: sec }, { text: "P5 floor", options: sec }, { text: "CVaR5", options: sec }, { text: "µ−σ", options: sec }],
 [{ text: "Shallow local optimum", options: cell }, { text: "0.0696", options: cell }, { text: "7.0%", options: cell }, { text: "0.0603", options: cell }, { text: "0.0532", options: cell }, { text: "0.0647", options: cell }],
 [{ text: "Nominal θ* (this work)", options: cell }, { text: "0.0795", options: cell }, { text: "2.5%", options: cell }, { text: "0.0758", options: cell }, { text: "0.0744", options: cell }, { text: "0.0775", options: cell }],
 [{ text: "Mean-variance corrected", options: cell }, { text: "0.0805", options: cell }, { text: "2.5%", options: cell }, { text: "0.0767", options: cell }, { text: "0.0753", options: cell }, { text: "0.0785", options: cell }],
 [{ text: "CVaR-corrected θ_rob (this work)", options: our }, { text: "0.0839", options: our }, { text: "2.6%", options: our }, { text: "0.0797", options: our }, { text: "0.0781", options: our }, { text: "0.0817", options: our }],
], { x: 0.5, y: 3.6, w: 8.6, colW: [2.6, 1.1, 0.9, 1.3, 1.3, 1.4], border: { pt: 0.5, color: "B0BEC5" } });
s.addTable([
 [{ text: "(D) Correction reuse (3–6 MHz)", options: sec }, { text: "Recovery of attainable gain", options: sec }],
 [{ text: "Static correction vector", options: cell }, { text: "−65% mean (best single freq.: 32%)", options: cell }],
 [{ text: "PARL-ID policy", options: our }, { text: "reusable by construction (zero-shot)", options: our }],
], { x: 9.4, y: 3.6, w: 3.4, colW: [1.6, 1.8], border: { pt: 0.5, color: "B0BEC5" } });
txt(s, "† as reported in the previous paper on the precursor database (row split; MAE normalized). All other values computed by the released pipeline.",
    0.5, 6.9, 12.3, 0.4, 9, "90A4AE", { align: "left", italic: true });
p.writeFile({ fileName: "/tmp/ppt/PARL-ID_figures.pptx" }).then(() => console.log("done"));
