import { C, base, title, body, label, node, rule, circle } from "./theme.mjs";

export async function slide01(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, 1, "Passion topic");
  title(slide, ctx, "Neural networks turn messy signals into engineering insight.", 56, 88, 670, 156);
  body(slide, ctx, "What excites me is the practical loop: collect data, learn patterns, test the output, then improve the system. It feels like engineering because the model has to earn trust in the real world.", 60, 286, 610, 116, 22);
  label(slide, ctx, "3-minute technical presentation", 60, 424, 330, 26, C.coral);

  const stages = [
    ["raw signal", "sensor data\nimages\nmeasurements"],
    ["learned pattern", "features\nweights\nrepresentations"],
    ["decision", "forecast\nclassification\nrecommendation"],
  ];
  const x0 = 750;
  stages.forEach(([head, copy], i) => {
    const y = 126 + i * 142;
    node(slide, ctx, { x: x0, y, w: 330, h: 78, text: copy, fill: i === 1 ? C.teal2 : C.panel, line: i === 1 ? C.teal : C.rule, color: C.ink, size: 20, bold: i === 1 });
    label(slide, ctx, head, x0, y - 30, 220, 22, i === 1 ? C.teal : C.muted);
    if (i < stages.length - 1) {
      rule(slide, ctx, x0 + 164, y + 88, 2, 48, i === 0 ? C.teal : C.coral);
      circle(slide, ctx, x0 + 154, y + 132, 22, i === 0 ? C.teal : C.coral, i === 0 ? C.teal : C.coral);
    }
  });

  body(slide, ctx, "My angle: not magic, not just maths. A neural network is a structured way to learn from evidence.", 748, 552, 360, 70, 18, C.ink);
  return slide;
}
