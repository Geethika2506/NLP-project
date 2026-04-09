#!/usr/bin/env python3
"""Generate PDF of codebase for Turnitin submission."""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Preformatted
from reportlab.lib import colors
import os
from pathlib import Path
from datetime import datetime

# Output file
pdf_file = "RL-Dataset_Codebase.pdf"
doc = SimpleDocTemplate(
    pdf_file, 
    pagesize=A4, 
    topMargin=0.5*inch, 
    bottomMargin=0.5*inch, 
    leftMargin=0.75*inch, 
    rightMargin=0.75*inch
)

story = []
styles = getSampleStyleSheet()

# Custom styles
title_style = ParagraphStyle(
    'CustomTitle',
    parent=styles['Heading1'],
    fontSize=22,
    textColor=colors.HexColor('#1f4788'),
    spaceAfter=12,
    alignment=1
)

heading_style = ParagraphStyle(
    'CustomHeading',
    parent=styles['Heading2'],
    fontSize=12,
    textColor=colors.HexColor('#2e5c8a'),
    spaceAfter=6,
    spaceBefore=12
)

code_style = ParagraphStyle(
    'CodeStyle',
    parent=styles['Normal'],
    fontName='Courier',
    fontSize=7,
    textColor=colors.HexColor('#333333'),
    spaceAfter=0,
    leading=8
)

# Title page
story.append(Paragraph("RL-Dataset Codebase", title_style))
story.append(Paragraph("Alignment Drift in Encoder-Decoder Models", styles['Heading2']))
story.append(Spacer(1, 0.2*inch))
story.append(Paragraph("Complete Source Code and Documentation", styles['Normal']))
story.append(Paragraph(f"Submitted for Turnitin<br/>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
story.append(Spacer(1, 0.3*inch))
story.append(PageBreak())

# README
story.append(Paragraph("README.md", heading_style))
try:
    with open('README.md', 'r', encoding='utf-8') as f:
        readme_text = f.read()
        story.append(Preformatted(readme_text, code_style))
except Exception as e:
    story.append(Paragraph(f"Error reading README: {e}", styles['Normal']))

story.append(Spacer(1, 0.2*inch))
story.append(PageBreak())

# requirements.txt
story.append(Paragraph("requirements.txt - Dependencies", heading_style))
try:
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        req_text = f.read()
        story.append(Preformatted(req_text, code_style))
except Exception as e:
    story.append(Paragraph(f"Error reading requirements: {e}", styles['Normal']))

story.append(Spacer(1, 0.2*inch))
story.append(PageBreak())

# Python files
python_files = [
    ('preprocessing.py', 'Data Preprocessing - Load JSON, tokenize, save tensors'),
    ('inference.py', 'Inference - Load models, generate responses, extract attention'),
    ('annotate.py', 'Annotation - Safety classification of model outputs'),
    ('features.py', 'Feature Extraction - Compute alignment metrics'),
    ('evaluate.py', 'Evaluation - Generate plots and statistical tests'),
    ('app.py', 'Web Interface - Gradio interactive app'),
]

for py_file, description in python_files:
    if Path(py_file).exists():
        story.append(Paragraph(f"{py_file}", heading_style))
        story.append(Paragraph(description, styles['Italic']))
        story.append(Spacer(1, 0.1*inch))
        
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # For large files, include first part
                max_lines = 300
                lines = content.split('\n')
                if len(lines) > max_lines:
                    content = '\n'.join(lines[:max_lines]) + f"\n\n[... {len(lines) - max_lines} more lines ...]"
                
                story.append(Preformatted(content, code_style))
        except Exception as e:
            story.append(Paragraph(f"Error reading {py_file}: {e}", styles['Normal']))
        
        story.append(Spacer(1, 0.15*inch))
        story.append(PageBreak())

# Build PDF
try:
    doc.build(story)
    file_size = os.path.getsize(pdf_file) / (1024*1024)  # MB
    abs_path = os.path.abspath(pdf_file)
    print(f"✅ PDF created successfully!")
    print(f"📄 File: {pdf_file}")
    print(f"📊 Size: {file_size:.1f} MB")
    print(f"📍 Location: {abs_path}")
except Exception as e:
    print(f"❌ Error creating PDF: {e}")
