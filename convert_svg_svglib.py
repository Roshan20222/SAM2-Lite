import os
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF

def convert_svg_to_pdf_svglib():
    """
    Finds all SVG files in the current directory and converts them to PDF using svglib.
    """
    current_directory = os.getcwd()
    for filename in os.listdir(current_directory):
        if filename.endswith(".svg"):
            svg_path = os.path.join(current_directory, filename)
            pdf_path = os.path.join(current_directory, filename.replace(".svg", ".pdf"))
            print(f"Converting {svg_path} to {pdf_path}...")
            try:
                drawing = svg2rlg(svg_path)
                renderPDF.drawToFile(drawing, pdf_path)
                print(f"Successfully converted {filename}")
            except Exception as e:
                print(f"Could not convert {filename}. Error: {e}")

if __name__ == "__main__":
    print("Starting SVG to PDF conversion with svglib...")
    try:
        import svglib
        import reportlab
    except ImportError:
        print("The 'svglib' or 'reportlab' library is not installed.")
        print("Please run 'pip install svglib reportlab' in your terminal.")
        exit()
    
    convert_svg_to_pdf_svglib()
    print("Conversion process finished.")
