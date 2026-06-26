import { C, base, title, body, label, node, rule } from "./theme.mjs";

export async function slide04(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, 4, "Use cases");
  title(slide, ctx, "You might wonder: what are neural networks actually used for?", 56, 78, 900, 110);
  body(slide, ctx, "They are useful when the input is too complex for hand-written rules, but there is enough data for the model to learn patterns.", 60, 190, 790, 52, 20);

  const rows = [
    ["Autonomy", "Tesla-style self-driving", "cameras / radar / lidar", "lanes, objects, motion prediction"],
    ["Language", "ChatGPT-style models", "text tokens + context", "summaries, answers, code, reasoning"],
    ["Vision", "image recognition", "photos, scans, site images", "classify, detect, inspect defects"],
    ["Engineering", "forecasting + monitoring", "sensor streams / simulations", "risk, anomalies, optimisation"],
  ];
  const x = 74;
  const y = 300;
  rows.forEach((row, r) => {
    const yy = y + r * 76;
    node(slide, ctx, { x, y: yy, w: 176, h: 54, text: row[0], fill: r === 0 ? C.coral : r === 1 ? C.teal2 : r === 2 ? C.gold : C.panel2, line: "transparent", color: r === 2 ? C.paper : C.white, bold: true, size: 18 });
    label(slide, ctx, row[1], x + 210, yy - 2, 280, 22, C.teal);
    body(slide, ctx, row[2], x + 210, yy + 22, 260, 30, 17, C.ink);
    rule(slide, ctx, x + 500, yy + 26, 54, 2, C.rule);
    body(slide, ctx, row[3], x + 590, yy + 10, 390, 38, 18, C.ink);
    rule(slide, ctx, x, yy + 64, 1030, 1, C.rule);
  });

  body(slide, ctx, "That is why they matter: the same core idea can understand roads, language, images, and engineering systems.", 76, 602, 940, 52, 20, C.ink);
  return slide;
}
