// Build the student guide: node build.js -> ../PINN_DFN_LiSPAN_student_guide.docx
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, AlignmentType, HeadingLevel, Footer, Header, PageNumber, TableOfContents,
  BorderStyle,
} = require("docx");
const h = require("./helpers");
const { FONT, INK, INK2 } = h;

const title = [
  new Paragraph({ spacing: { before: 2400, after: 240 }, children: [new TextRun({ text: "Physics-Informed Neural Networks", font: FONT, size: 56, bold: true, color: "1F4E8C" })] }),
  new Paragraph({ spacing: { after: 240 }, children: [new TextRun({ text: "for Battery Models", font: FONT, size: 56, bold: true, color: "1F4E8C" })] }),
  new Paragraph({ spacing: { after: 600 }, border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: "2A78D6", space: 8 } },
    children: [new TextRun({ text: "From the lithium-ion DFN model to lithium-SPAN cells: a student guide to the project and its results", font: FONT, size: 30, color: INK2 })] }),
  new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text: "Project: PINN-DFN (Balbuena group, Department of Chemical Engineering, Texas A&M University)", font: FONT, size: 22, color: INK })] }),
  new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text: "Project lead: Carlos Guerrero", font: FONT, size: 22, color: INK })] }),
  new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text: "Prepared for the undergraduate progress presentation, October 2026", font: FONT, size: 22, color: INK })] }),
  new Paragraph({ spacing: { before: 1800 }, children: [new TextRun({ text: "Written in plain language: theory, equations and analogies first, then the results. Numbers and figures are generated from the project repository (DFN_PINN_Project).", font: FONT, size: 20, italics: true, color: INK2 })] }),
  new Paragraph({ pageBreakBefore: true, spacing: { after: 240 }, children: [new TextRun({ text: "Contents", font: FONT, size: 32, bold: true, color: "1F4E8C" })] }),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  new Paragraph({ spacing: { before: 200 }, children: [new TextRun({ text: "(If the table of contents is empty, right-click it in Word and choose 'Update field'.)", font: FONT, size: 18, italics: true, color: INK2 })] }),
];

const buildBody = () => [
  ...require("./ch_intro")(h),
  ...require("./ch_models")(h),
  ...require("./ch_dfn")(h),
  ...require("./ch_lispan")(h),
  ...require("./ch_end")(h),
].flat(Infinity);   // boxes return [table, spacer]; allow them without spreading
buildBody();            // first pass: figure numbers for the {figNN} references
h.resetCounters();
const body = buildBody();

const doc = new Document({
  creator: "Carlos Guerrero / Claude",
  title: "PINNs for battery models: student guide",
  description: "Student guide to the PINN-DFN project (DFN and Li-SPAN)",
  features: { updateFields: true },
  styles: {
    default: { document: { run: { font: FONT, size: 22, color: INK } } },
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
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
      children: [new TextRun({ text: "PINNs for battery models - student guide", font: FONT, size: 16, color: INK2 })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18, color: INK2 })] })] }) },
    children: [...title, ...body],
  }],
});

const out = path.join(__dirname, "..", "PINN_DFN_LiSPAN_student_guide.docx");
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(out, buf); console.log("wrote", out, buf.length); });
