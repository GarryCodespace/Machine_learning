import { C, base, title, body, label, rule, circle } from "./theme.mjs";

export async function slide02(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, 2, "Representation");
  title(slide, ctx, "A neural network is layers of weighted connections, not a single graph.", 56, 78, 900, 112);
  body(slide, ctx, "Information moves forward through neurons. Training changes the connection weights so the final output gets closer to the target.", 60, 194, 760, 58, 20);
  ctx.addShape(slide, { x: 60, y: 566, w: 450, h: 84, fill: C.panel, line: ctx.line(C.rule, 1) });
  label(slide, ctx, "one layer in linear algebra", 84, 578, 230, 22, C.gold);
  body(slide, ctx, "z = W x + b   →   a = σ(z)", 84, 606, 350, 28, 22, C.ink);

  const layers = [
    { x: 150, label: "input features", dots: 4, fill: C.panel2 },
    { x: 365, label: "hidden layer", dots: 6, fill: C.teal2 },
    { x: 595, label: "hidden layer", dots: 5, fill: C.teal },
    { x: 815, label: "outputs", dots: 3, fill: C.coral },
  ];
  ctx.addShape(slide, { x: 212, y: 330, w: 700, h: 216, fill: "#0B2033", line: ctx.line(C.rule, 1) });
  label(slide, ctx, "dense weighted connections", 430, 555, 260, 22, C.gold);
  layers.forEach((layer, li) => {
    label(slide, ctx, layer.label, layer.x - 44, 300, 140, 22, li === 0 ? C.muted : C.teal);
    for (let i = 0; i < layer.dots; i += 1) {
      const y = 345 + i * 42 + (6 - layer.dots) * 21;
      circle(slide, ctx, layer.x, y, 22, layer.fill, li === 0 ? C.rule : layer.fill);
    }
  });
  for (let li = 0; li < layers.length - 1; li += 1) {
    const left = layers[li];
    const right = layers[li + 1];
    for (let i = 0; i < left.dots; i += 1) {
      const y1 = 356 + i * 42 + (6 - left.dots) * 21;
      for (let j = 0; j < right.dots; j += 1) {
        if ((i + j + li) % 2 !== 0) continue;
        const y2 = 356 + j * 42 + (6 - right.dots) * 21;
        const midX = left.x + 54;
        const midW = right.x - left.x - 108;
        const midY = (y1 + y2) / 2;
        rule(slide, ctx, midX, midY, midW, 1, li === 1 ? "#3FB6C1" : "#39536E");
      }
    }
  }

  const callouts = [
    ["W", "weights: how strongly inputs connect"],
    ["b", "bias: shifts the decision boundary"],
    ["σ", "activation: adds non-linearity"],
  ];
  callouts.forEach(([h, c], i) => {
    const x = 940;
    const y = 322 + i * 84;
    label(slide, ctx, h, x, y, 150, 22, i === 1 ? C.coral : C.teal);
    body(slide, ctx, c, x, y + 24, 210, 48, 17, C.ink);
  });
  return slide;
}
