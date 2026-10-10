// Build the advisor summary: node build.js -> ../PINN_LiSPAN_advisor_summary.docx  (numbers from ../numbers.json)
const fs = require("fs");
const path = require("path");
const { Document, Packer, Paragraph, TextRun, AlignmentType, Footer, Header, PageNumber, BorderStyle } = require("docx");
const h = require("./helpers");
const { FONT, INK, INK2 } = h;
const N = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "numbers.json"), "utf8"));

const title = [
  new Paragraph({ spacing: { before: 0, after: 120 }, children: [new TextRun({ text: "From a classic battery model to lithium-SPAN with a physics-informed neural network", font: FONT, size: 40, bold: true, color: "1F4E8C" })] }),
  new Paragraph({ spacing: { after: 160 }, border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: "2A78D6", space: 6 } },
    children: [new TextRun({ text: "Summary of results for the advisor meeting", font: FONT, size: 26, color: INK2 })] }),
  new Paragraph({ spacing: { after: 240 }, children: [new TextRun({ text: "PINN-DFN project, Balbuena group, Texas A&M University · Carlos Guerrero · October 2026", font: FONT, size: 19, color: INK2 })] }),
];

const buildBody = () => [
  ...require("./ch1_background")(h, N),
  ...require("./ch3_results")(h, N),
  ...require("./ch4_chem")(h, N),
  ...require("./ch5_conclusions")(h, N),
].flat(Infinity);
buildBody();            // first pass: figure and table numbers for the cross-references
h.resetCounters();
const body = buildBody();

const doc = new Document({
  creator: "Carlos Guerrero / Claude",
  title: "PINN for DFN and Li-SPAN: summary for the advisor meeting",
  description: "Simple-language summary of the PINN-DFN and Li-SPAN results, for the advisor meeting",
  styles: {
    default: { document: { run: { font: FONT, size: 21, color: INK } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: FONT, color: "1F4E8C" }, paragraph: { spacing: { before: 120, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, font: FONT, color: "2A78D6" }, paragraph: { spacing: { before: 260, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, font: FONT, color: INK }, paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 2 } },
    ],
  },
  numbering: h.numbering,
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1152, right: 1152, bottom: 1152, left: 1152 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
      children: [new TextRun({ text: "PINN: classic DFN to Li-SPAN - summary for the advisor meeting", font: FONT, size: 16, color: INK2 })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18, color: INK2 })] })] }) },
    children: [...title, ...body],
  }],
});

const out = path.join(__dirname, "..", "PINN_LiSPAN_advisor_summary.docx");
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(out, buf); console.log("wrote", out, buf.length); });
