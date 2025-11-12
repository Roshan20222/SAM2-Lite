import os
from cairosvg import svg2pdf

def convert_svg_to_pdf():
    """
    Finds all SVG files in the current directory and converts them to PDF.
    """
    current_directory = os.getcwd()
    for filename in os.listdir(current_directory):
        if filename.endswith(".svg"):
            svg_path = os.path.join(current_directory, filename)
            pdf_path = os.path.join(current_directory, filename.replace(".svg", ".pdf"))
            print(f"Converting {svg_path} to {pdf_path}...")
            try:
                svg2pdf(url=svg_path, write_to=pdf_path)
                print(f"Successfully converted {filename}")
            except Exception as e:
                print(f"Could not convert {filename}. Error: {e}")

if __name__ == "__main__":
    print("Starting SVG to PDF conversion...")
    # First, ensure the required library is installed.
    try:
        import cairosvg
    except ImportError:
        print("The 'cairosvg' library is not installed.")
        print("Please run 'pip install cairosvg' in your terminal.")
        # You might need to install GTK+ for cairosvg to work on Windows.
        # See https://www.cairosvg.org/documentation/
        exit()
    
    convert_svg_to_pdf()
    print("Conversion process finished.")
