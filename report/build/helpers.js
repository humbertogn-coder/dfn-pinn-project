// Helpers for the student guide (docx-js). Inline markup in text: **bold**, *italic*, _{sub}, ^{sup}, `code`.
const fs = require("fs");
const path = require("path");
const {
  Paragraph, TextRun, HeadingLevel, AlignmentType, ImageRun, Table, TableRow, TableCell, WidthType,
  ShadingType, BorderStyle, PageBreak, LevelFormat,
} = require("docx");

const FIG = path.join(__dirname, "..", "figures");
const EQM = JSON.parse(fs.readFileSync(path.join(__dirname, "eq_manifest.json"), "utf8"));
const FONT = "Calibri";
const INK = "0B0B0B", INK2 = "52514E";
const CONTENT_W = 9360; // DXA, US Letter with 1" margins

let figNo = 0, eqNo = 0, tabNo = 0;
const figLabels = {};          // "fig05" -> figure number (filled in the first build pass)
function resetCounters() { figNo = 0; eqNo = 0; tabNo = 0; olCount = 0; }
function resolve(text) { return text.replace(/\{(fig\d+)\}/g, (m, k) => (figLabels[k] !== undefined ? String(figLabels[k]) : "?")); }

function runs(text, base = {}) {
  text = resolve(text);
  // tokenizer for **bold**, *italic*, _{sub}, ^{sup}, `code`
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|_\{[^}]*\}|\^\{[^}]*\}|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...base }));
    const tok = m[0];
    if (tok.startsWith("**")) out.push(new TextRun({ text: tok.slice(2, -2), bold: true, font: FONT, ...base }));
    else if (tok.startsWith("*")) out.push(new TextRun({ text: tok.slice(1, -1), italics: true, font: FONT, ...base }));
    else if (tok.startsWith("_{")) out.push(new TextRun({ text: tok.slice(2, -1), subScript: true, font: FONT, ...base }));
    else if (tok.startsWith("^{")) out.push(new TextRun({ text: tok.slice(2, -1), superScript: true, font: FONT, ...base }));
    else if (tok.startsWith("`")) out.push(new TextRun({ text: tok.slice(1, -1), font: "Consolas", size: 19, ...base }));
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...base }));
  return out;
}

const P = (text, opts = {}) => new Paragraph({ children: runs(text, opts.run || {}), spacing: { after: 120, line: 288 },
  alignment: opts.align || AlignmentType.LEFT, ...(opts.para || {}) });
const H1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text, font: FONT })], pageBreakBefore: true });
const H2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text, font: FONT })] });
const H3 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun({ text, font: FONT })] });
const BR = () => new Paragraph({ children: [new PageBreak()] });

const UL = (items, level = 0) => items.map((t) => new Paragraph({ numbering: { reference: "bullets", level },
  children: runs(t), spacing: { after: 80, line: 276 } }));
const OL = (items, ref = "numbers") => items.map((t) => new Paragraph({ numbering: { reference: ref, level: 0 },
  children: runs(t), spacing: { after: 80, line: 276 } }));

function EQ(key, label = true) {
  const e = EQM[key];
  if (!e) throw new Error("missing equation " + key);
  let wIn = (e.w / e.dpi) * (11.5 / 15);
  let hIn = (e.h / e.dpi) * (11.5 / 15);
  const maxW = 6.0;
  if (wIn > maxW) { hIn *= maxW / wIn; wIn = maxW; }
  const children = [new ImageRun({ type: "png", data: fs.readFileSync(e.file),
    transformation: { width: Math.round(wIn * 96), height: Math.round(hIn * 96) },
    altText: { title: key, description: "equation " + key, name: key } })];
  if (label) { eqNo += 1; children.push(new TextRun({ text: `\t(${eqNo})`, font: FONT, color: INK2 })); }
  return new Paragraph({ children, alignment: AlignmentType.CENTER, spacing: { before: 120, after: 160 } });
}

function FIGURE(file, caption, widthIn = 6.3) {
  figNo += 1;
  figLabels[file.slice(0, 5)] = figNo;
  const p = path.join(FIG, file);
  const { w, h } = pngSize(p);
  const hIn = widthIn * h / w;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 160, after: 60 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: fs.readFileSync(p),
        transformation: { width: Math.round(widthIn * 96), height: Math.round(hIn * 96) },
        altText: { title: file, description: caption.replace(/\*/g, ""), name: file } })] }),
    new Paragraph({ spacing: { after: 200 }, children: [new TextRun({ text: `Figure ${figNo}. `, bold: true, font: FONT, size: 19, color: INK2 }),
      ...runs(caption, { size: 19, color: INK2 })] }),
  ];
}

function pngSize(p) {
  const b = fs.readFileSync(p);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "C9C8C3" };
const borders = { top: border, bottom: border, left: border, right: border };

function BOX(title, paras, fill = "EEF4FC", accent = "2A78D6") {
  // shaded single-cell table used for analogies, key ideas and warnings
  const left = { style: BorderStyle.SINGLE, size: 24, color: accent };
  const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  const kids = [new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: title, bold: true, font: FONT, color: accent })] })];
  for (const t of paras) {
    if (Array.isArray(t)) kids.push(...UL(t));
    else kids.push(P(t));
  }
  return [new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: [CONTENT_W],
    rows: [new TableRow({ cantSplit: true, children: [new TableCell({ width: { size: CONTENT_W, type: WidthType.DXA },
      borders: { top: none, bottom: none, right: none, left },
      shading: { fill, type: ShadingType.CLEAR, color: "auto" },
      margins: { top: 120, bottom: 80, left: 200, right: 200 }, children: kids })] })] }),
    new Paragraph({ spacing: { after: 120 }, children: [] })];
}
const ANALOGY = (title, paras) => BOX("Analogy: " + title, paras, "EEF4FC", "2A78D6");
const KEY = (title, paras) => BOX("Key idea: " + title, paras, "E8F6F0", "1BAF7A");
const CAUTION = (title, paras) => BOX("Be careful: " + title, paras, "FDF0EA", "EB6834");
const SLIDE = (paras) => BOX("For the slides", paras, "F6F6F4", "52514E");

function TABLE(header, rows, widths, caption) {
  tabNo += 1;
  const total = widths.reduce((a, b) => a + b, 0);
  const scale = CONTENT_W / total;
  const ws = widths.map((w) => Math.round(w * scale));
  ws[ws.length - 1] += CONTENT_W - ws.reduce((a, b) => a + b, 0);
  const mk = (cells, head) => new TableRow({ tableHeader: head, children: cells.map((c, i) => new TableCell({
    width: { size: ws[i], type: WidthType.DXA }, borders,
    shading: head ? { fill: "EEF4FC", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ children: runs(String(c), head ? { bold: true, size: 19 } : { size: 19 }) })] })) });
  const out = [];
  if (caption) out.push(new Paragraph({ keepNext: true, spacing: { before: 160, after: 80 },
    children: [new TextRun({ text: `Table ${tabNo}. `, bold: true, font: FONT, size: 19, color: INK2 }), ...runs(caption, { size: 19, color: INK2 })] }));
  out.push(new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: ws, rows: [mk(header, true), ...rows.map((r) => mk(r, false))] }));
  out.push(new Paragraph({ spacing: { after: 160 }, children: [] }));
  return out;
}

const numbering = { config: [
  { reference: "bullets", levels: [
    { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
    { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1440, hanging: 360 } } } }] },
  ...Array.from({ length: 30 }, (_, i) => ({ reference: i === 0 ? "numbers" : `numbers${i}`, levels: [
    { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] })),
] };

let olCount = 0;
const OLn = (items) => { olCount += 1; return OL(items, `numbers${olCount}`); };

module.exports = { resetCounters, P, H1, H2, H3, BR, UL, OL: OLn, EQ, FIGURE, ANALOGY, KEY, CAUTION, SLIDE, TABLE, numbering, runs, FONT, INK, INK2 };
