#!/usr/bin/env python3
import os
import sys

try:
    import fitz
except ImportError:
    print("Installing PyMuPDF...")
    os.system("pip install pymupdf -q")
    import fitz

def html_to_pdf(html_file, pdf_file):
    """Convert HTML file to PDF using PyMuPDF."""
    
    if not os.path.exists(html_file):
        return False
    
    try:
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Create PDF document
        doc = fitz.open()
        page = doc.new_page()
        
        # Extract and clean text from HTML
        import re
        text = re.sub('<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL)
        text = re.sub('<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub('<[^>]+>', '', text)
        
        # Decode HTML entities
        replacements = {
            '&mdash;': '—',
            '&ndash;': '–',
            '&nbsp;': ' ',
            '&lt;': '<',
            '&gt;': '>',
            '&amp;': '&',
            '&quot;': '"',
            '&apos;': "'",
            '&kappa;': 'κ',
            '&Delta;': 'Δ',
            '&times;': '×',
            '&le;': '≤',
            '&ge;': '≥',
            '&#8804;': '≤',
            '&#8805;': '≥',
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        
        # Clean up whitespace
        lines = text.split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        text = '\n'.join(lines)
        
        # Insert text into PDF (split by page if needed)
        y_pos = 36
        page_height = 792
        margins = 36
        line_height = 12
        
        for line in lines[:500]:  # Limit to 500 lines per file
            if y_pos > page_height - margins:
                page = doc.new_page()
                y_pos = margins
            
            if len(line) > 100:
                line = line[:97] + '...'
            
            page.insert_text((margins, y_pos), line, fontsize=9, fontname="helv")
            y_pos += line_height
        
        doc.save(pdf_file)
        doc.close()
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == '__main__':
    files = [
        ('RESEARCH_DATA_TABLES.html', 'RESEARCH_DATA_TABLES.pdf'),
        ('PAPER_RESULTS_SECTION.html', 'PAPER_RESULTS_SECTION.pdf'),
        ('PAPER_ABSTRACT_CONCLUSION.html', 'PAPER_ABSTRACT_CONCLUSION.pdf'),
    ]
    
    print("📄 Creating PDFs from HTML files...\n")
    
    for html_file, pdf_file in files:
        print(f"   Converting {html_file}...", end=" ", flush=True)
        if html_to_pdf(html_file, pdf_file):
            if os.path.exists(pdf_file):
                size = os.path.getsize(pdf_file) / 1024
                print(f"✅ ({size:.1f} KB)")
            else:
                print("❌ (file not created)")
        else:
            print("❌ (error)")
    
    print("\n" + "="*50)
    print("✅ PDF creation complete!")
    print("="*50)
    
    for _, pdf_file in files:
        if os.path.exists(pdf_file):
            size = os.path.getsize(pdf_file) / 1024
            print(f"\n   ✓ {pdf_file} ({size:.1f} KB)")
