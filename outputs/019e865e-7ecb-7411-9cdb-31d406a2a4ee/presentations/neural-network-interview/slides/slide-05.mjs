import { C, base, title, body, label, node, rule } from "./theme.mjs";

export async function slide05(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, 5, "Delivery mindset");
  title(slide, ctx, "A model is only useful when it is validated, explainable enough, and monitored.", 56, 76, 880, 116);
  body(slide, ctx, "My interest in neural networks is not just building models. It is building systems that can be trusted in an engineering context.", 60, 196, 760, 58, 20);

  const stages = [
    ["data quality", "representative, labelled,\nchecked for bias"],
    ["model validation", "test on unseen data,\nmeasure failure modes"],
    ["human judgement", "explain enough to support\nengineering decisions"],
    ["deployment monitoring", "track drift, feedback,\nand real-world outcomes"],
  ];
  stages.forEach(([h, c], i) => {
    const x = 96 + i * 285;
    node(slide, ctx, { x, y: 330, w: 220, h: 112, text: c, fill: i === 3 ? C.teal2 : C.panel, line: i === 3 ? C.teal : C.rule, color: C.ink, size: 17 });
    label(slide, ctx, h, x, 292, 220, 24, i === 3 ? C.coral : C.teal);
    if (i < stages.length - 1) {
      rule(slide, ctx, x + 226, 384, 52, 3, i === 1 ? C.coral : C.rule);
    }
  });

  ctx.addShape(slide, { x: 90, y: 506, w: 1000, h: 86, fill: C.teal, line: ctx.line(C.teal, 1) });
  body(slide, ctx, "For me, neural networks are exciting because they combine mathematics, software, data, and real-world judgement - exactly the kind of multidisciplinary thinking I want to grow as an engineer.", 118, 522, 940, 56, 20, C.white);
  return slide;
}
