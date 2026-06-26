import { C, base, title, body, label, node, rule, circle } from "./theme.mjs";

export async function slide03(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, 3, "Learning loop");
  title(slide, ctx, "Training is an engineering feedback loop: predict, measure error, update.", 56, 76, 870, 110);
  body(slide, ctx, "Backpropagation sounds abstract, but the mindset is familiar: compare expected and actual performance, then tune the system to reduce error.", 60, 192, 760, 58, 20);

  const loop = [
    { text: "1\npredict", x: 170, y: 330, fill: C.panel, color: C.ink },
    { text: "2\ncalculate loss", x: 390, y: 250, fill: C.coral, color: C.white },
    { text: "3\nbackpropagate", x: 620, y: 330, fill: C.panel2, color: C.ink },
    { text: "4\nupdate weights", x: 390, y: 450, fill: C.teal, color: C.paper },
  ];
  loop.forEach((d) => node(slide, ctx, { x: d.x, y: d.y, w: 150, h: 78, text: d.text, fill: d.fill, line: d.fill, color: d.color, bold: true }));
  rule(slide, ctx, 326, 365, 58, 3, C.rule);
  rule(slide, ctx, 542, 292, 68, 3, C.rule);
  rule(slide, ctx, 542, 486, 68, 3, C.rule);
  rule(slide, ctx, 326, 486, 58, 3, C.rule);
  label(slide, ctx, "forward pass", 322, 334, 130, 22, C.teal);
  label(slide, ctx, "gradient", 542, 306, 120, 22, C.coral);
  label(slide, ctx, "repeat", 348, 542, 120, 22, C.muted);

  label(slide, ctx, "loss falls as feedback improves the fit", 878, 278, 330, 22, C.teal);
  const chartX = 850;
  const chartY = 330;
  rule(slide, ctx, chartX, chartY + 150, 340, 2, C.ink);
  rule(slide, ctx, chartX, chartY, 2, 152, C.ink);
  const bars = [136, 104, 76, 54, 38, 30];
  bars.forEach((h, i) => {
    const x = chartX + 38 + i * 48;
    ctx.addShape(slide, { x, y: chartY + 150 - h, w: 28, h, fill: i < 2 ? C.coral : i < 4 ? C.gold : C.teal, line: ctx.line() });
  });
  label(slide, ctx, "epochs", chartX + 250, chartY + 160, 90, 22, C.muted);
  label(slide, ctx, "error", chartX - 70, chartY - 44, 70, 22, C.muted);
  circle(slide, ctx, 1080, 514, 16, C.teal, C.teal);
  body(slide, ctx, "The goal is not perfection; it is generalisation to new examples.", 850, 545, 350, 42, 18, C.ink);
  return slide;
}
