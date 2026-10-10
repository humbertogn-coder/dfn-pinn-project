// Build the results summary: node build.js -> ../PINN_LiSPAN_results_summary.docx
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, AlignmentType, Footer, Header, PageNumber, TableOfContents, BorderStyle,
} = require("docx");
const h = require("./helpers");
const { FONT, INK, INK2 } = h;

const title = [
  new Paragraph({ spacing: { before: 2200, after: 240 }, children: [new TextRun({ text: "A PINN for a Lithium-SPAN Cell Model", font: FONT, size: 52, bold: true, color: "1F4E8C" })] }),
  new Paragraph({ spacing: { after: 600 }, border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: "2A78D6", space: 8 } },
    children: [new TextRun({ text: "Results, analysis and limitations: a summary for the advisor review before writing the manuscript", font: FONT, size: 30, color: INK2 })] }),
  new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text: "Project: PINN-DFN, Li-SPAN part (Balbuena group, Department of Chemical Engineering, Texas A&M University)", font: FONT, size: 22, color: INK })] }),
  new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text: "Project lead: Carlos Guerrero", font: FONT, size: 22, color: INK })] }),
  new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text: "Status of the work: October 9, 2026 (paper steps 1-5 completed)", font: FONT, size: 22, color: INK })] }),
  new Paragraph({ spacing: { before: 1600 }, children: [new TextRun({ text: "Written in the plain language of the student guide (October 2026). All numbers and figures are generated from the project repository (DFN_PINN_Project) by scripts/paper_build.py; see the Appendices.", font: FONT, size: 20, italics: true, color: INK2 })] }),
  new Paragraph({ pageBreakBefore: true, spacing: { after: 240 }, children: [new TextRun({ text: "Contents", font: FONT, size: 32, bold: true, color: "1F4E8C" })] }),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  new Paragraph({ spacing: { before: 200 }, children: [new TextRun({ text: "(If the table of contents is empty, right-click it in Word and choose 'Update field'.)", font: FONT, size: 18, italics: true, color: INK2 })] }),
];

const buildBody = () => [
  ...require("./ch0_summary")(h),
  ...require("./ch1_context")(h),
  ...require("./ch2_method")(h),
  ...require("./ch3_phases")(h),
  ...require("./ch4_results")(h),
  ...require("./ch5_analysis")(h),
  ...require("./ch8_end")(h),
].flat(Infinity);
buildBody();            // first pass: figure and table numbers for the cross-references
h.resetCounters();
const body = buildBody();

const doc = new Document({
  creator: "Carlos Guerrero / Claude",
  title: "A PINN for a Li-SPAN cell model: results summary",
  description: "Results, analysis and limitations of the Li-SPAN PINN work, for the advisor review",
  features: { updateFields: true },
  styles: {
    default: { document: { run: { font: FONT, size: 21, color: INK } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 34, bold: true, font: FONT, color: "1F4E8C" }, paragraph: { spacing: { before: 120, after: 240 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 27, bold: true, font: FONT, color: "2A78D6" }, paragraph: { spacing: { before: 300, after: 140 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 23, bold: true, font: FONT, color: INK }, paragraph: { spacing: { before: 220, after: 100 }, outlineLevel: 2 } },
    ],
  },
  numbering: h.numbering,
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1296, right: 1296, bottom: 1296, left: 1296 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
      children: [new TextRun({ text: "Li-SPAN PINN - results summary for the advisor review", font: FONT, size: 16, color: INK2 })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18, color: INK2 })] })] }) },
    children: [...title, ...body],
  }],
});

const out = path.join(__dirname, "..", "PINN_LiSPAN_results_summary.docx");
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(out, buf); console.log("wrote", out, buf.length); });
