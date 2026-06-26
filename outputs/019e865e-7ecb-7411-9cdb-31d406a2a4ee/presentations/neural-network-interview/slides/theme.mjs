export const C = {
  paper: "#07111F",
  paper2: "#0E1A2A",
  panel: "#111D2F",
  panel2: "#17263B",
  ink: "#EAF2FF",
  muted: "#91A1B5",
  rule: "#26384F",
  teal: "#4DD3D8",
  teal2: "#2A8B96",
  coral: "#FF7A59",
  olive: "#B7D36A",
  gold: "#F2C14E",
  white: "#FFFFFF",
};

export function base(slide, ctx, n, kicker) {
  ctx.addShape(slide, { x: 0, y: 0, w: ctx.W, h: ctx.H, fill: C.paper, line: ctx.line() });
  ctx.addShape(slide, { x: 56, y: 42, w: 10, h: 10, fill: C.teal, line: ctx.line(), name: `kicker-${n}-marker` });
  ctx.addText(slide, {
    x: 74, y: 34, w: 250, h: 26, text: kicker.toUpperCase(), size: 13, bold: true,
    color: C.teal, typeface: ctx.fonts.mono, valign: "middle", name: `kicker-${n}-label`,
  });
  ctx.addShape(slide, { x: 56, y: 662, w: 1168, h: 1, fill: C.rule, line: ctx.line() });
  ctx.addText(slide, { x: 1030, y: 668, w: 194, h: 18, text: `Neural networks / 0${n}`, size: 10, color: C.muted, align: "right" });
}

export function title(slide, ctx, text, x = 56, y = 76, w = 720, h = 112) {
  return ctx.addText(slide, { x, y, w, h, text, size: 42, bold: true, color: C.ink, typeface: ctx.fonts.title });
}

export function body(slide, ctx, text, x, y, w, h, size = 20, color = C.ink) {
  return ctx.addText(slide, { x, y, w, h, text, size, color, insets: { left: 0, right: 0, top: 2, bottom: 2 } });
}

export function label(slide, ctx, text, x, y, w, h, color = C.muted) {
  return ctx.addText(slide, { x, y, w, h, text, size: 13, color, typeface: ctx.fonts.mono, valign: "middle" });
}

export function node(slide, ctx, { x, y, w, h, text, fill = C.panel, line = C.rule, color = C.ink, size = 17, bold = false }) {
  const s = ctx.addShape(slide, { x, y, w, h, fill, line: ctx.line(line, 1) });
  s.text = text;
  s.text.fontSize = size;
  s.text.color = color;
  s.text.bold = bold;
  s.text.typeface = ctx.fonts.body;
  s.text.alignment = "center";
  s.text.verticalAlignment = "middle";
  s.text.insets = { left: 12, right: 12, top: 10, bottom: 10 };
  return s;
}

export function rule(slide, ctx, x, y, w, h = 2, fill = C.rule) {
  return ctx.addShape(slide, { x, y, w, h, fill, line: ctx.line() });
}

export function circle(slide, ctx, x, y, d, fill, line = C.rule) {
  return ctx.addShape(slide, { x, y, w: d, h: d, geometry: "ellipse", fill, line: ctx.line(line, 1) });
}
