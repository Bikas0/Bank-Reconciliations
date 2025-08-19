import os
import io
from pdf2image import convert_from_path
import fitz  # PyMuPDF
from PIL import Image

def pdf_to_images_pdf2image(pdf_path, output_folder="images", dpi=300, format="PNG"):
    """
    Convert PDF to images using pdf2image library
    
    Args:
        pdf_path (str): Path to the PDF file
        output_folder (str): Folder to save images
        dpi (int): Resolution of output images
        format (str): Image format (PNG, JPEG, etc.)
    
    Returns:
        list: List of saved image paths
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        # Convert PDF to images
        pages = convert_from_path(pdf_path, dpi=dpi)
        
        saved_images = []
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        for i, page in enumerate(pages):
            # Save each page as an image
            image_path = os.path.join(output_folder, f"{pdf_name}_page_{i+1}.{format.lower()}")
            page.save(image_path, format)
            saved_images.append(image_path)
            print(f"Saved: {image_path}")
        
        return saved_images
        
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return []

def pdf_to_images_pymupdf(pdf_path, output_folder="images", zoom=2.0, format="PNG"):
    """
    Convert PDF to images using PyMuPDF (faster, no external dependencies)
    
    Args:
        pdf_path (str): Path to the PDF file
        output_folder (str): Folder to save images
        zoom (float): Zoom factor for image quality (2.0 = 200%)
        format (str): Image format (PNG, JPEG, etc.)
    
    Returns:
        list: List of saved image paths
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        # Open PDF
        pdf_document = fitz.open(pdf_path)
        saved_images = []
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # Create transformation matrix for zoom
        mat = fitz.Matrix(zoom, zoom)
        
        for page_num in range(pdf_document.page_count):
            # Get page
            page = pdf_document[page_num]
            
            # Render page to image
            pix = page.get_pixmap(matrix=mat)
            
            # Save image
            image_path = os.path.join(output_folder, f"{page_num+1}.{format.lower()}")
            
            if format.upper() == "PNG":
                pix.save(image_path)
            else:
                # Convert to PIL Image for other formats
                img_data = pix.tobytes("ppm")
                img = Image.open(io.BytesIO(img_data))
                img.save(image_path, format.upper())
            
            saved_images.append(image_path)
            print(f"Saved: {image_path}")
        
        pdf_document.close()
        return saved_images
        
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return []

def convert_specific_pages(pdf_path, pages, output_folder="images", method="pymupdf"):
    """
    Convert specific pages of PDF to images
    
    Args:
        pdf_path (str): Path to the PDF file
        pages (list): List of page numbers to convert (1-indexed)
        output_folder (str): Folder to save images
        method (str): Method to use ('pymupdf' or 'pdf2image')
    
    Returns:
        list: List of saved image paths
    """
    os.makedirs(output_folder, exist_ok=True)
    saved_images = []
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    if method == "pymupdf":
        try:
            pdf_document = fitz.open(pdf_path)
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom
            
            for page_num in pages:
                if 1 <= page_num <= pdf_document.page_count:
                    page = pdf_document[page_num - 1]  # Convert to 0-indexed
                    pix = page.get_pixmap(matrix=mat)
                    
                    image_path = os.path.join(output_folder, f"{page_num}.png")
                    pix.save(image_path)
                    saved_images.append(image_path)
                    print(f"Saved: {image_path}")
                else:
                    print(f"Page {page_num} is out of range")
            
            pdf_document.close()
            
        except Exception as e:
            print(f"Error: {e}")
    
    elif method == "pdf2image":
        try:
            # Convert only specific pages
            images = convert_from_path(pdf_path, first_page=min(pages), last_page=max(pages))
            
            for i, page_num in enumerate(pages):
                if i < len(images):
                    image_path = os.path.join(output_folder, f"{i}.png")
                    images[i].save(image_path, "PNG")
                    saved_images.append(image_path)
                    print(f"Saved: {image_path}")
        
        except Exception as e:
            print(f"Error: {e}")
    
    return saved_images

def batch_convert_pdfs(pdf_folder, output_folder="batch_images", method="pymupdf"):
    """
    Convert all PDFs in a folder to images
    
    Args:
        pdf_folder (str): Folder containing PDF files
        output_folder (str): Folder to save all images
        method (str): Method to use ('pymupdf' or 'pdf2image')
    """
    os.makedirs(output_folder, exist_ok=True)
    
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        print(f"\nProcessing: {pdf_file}")
        
        if method == "pymupdf":
            pdf_to_images_pymupdf(pdf_path, output_folder)
        else:
            pdf_to_images_pdf2image(pdf_path, output_folder)

def convert_pdf_to_images(pdf_path, output_folder="pdf_images", method="pymupdf", 
                         zoom=2.0, dpi=300, pages=None, format="PNG"):
    """
    Main function to convert PDF to images - can be called from other scripts
    
    Args:
        pdf_path (str): Path to the PDF file
        output_folder (str): Folder to save images (default: "pdf_images")
        method (str): Conversion method - "pymupdf" or "pdf2image" (default: "pymupdf")
        zoom (float): Zoom factor for PyMuPDF (default: 2.0)
        dpi (int): DPI for pdf2image (default: 300)
        pages (list): List of specific pages to convert (1-indexed), None for all pages
        format (str): Image format (default: "PNG")
    
    Returns:
        list: List of saved image paths
    
    Example:
        # Convert entire PDF
        images = convert_pdf_to_images("my_document.pdf")
        
        # Convert specific pages with high quality
        images = convert_pdf_to_images("my_document.pdf", pages=[1, 2, 3], zoom=3.0)
        
        # Convert using pdf2image method
        images = convert_pdf_to_images("my_document.pdf", method="pdf2image", dpi=400)
    """
    
    # Check if PDF file exists
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return []
    
    print(f"📄 Converting PDF: {os.path.basename(pdf_path)}")
    print(f"📂 Output folder: {output_folder}")
    print(f"🔧 Method: {method}")
    
    try:
        if pages:
            # Convert specific pages
            print(f"📋 Converting pages: {pages}")
            saved_images = convert_specific_pages(pdf_path, pages, output_folder, method)
        else:
            # Convert all pages
            print("📋 Converting all pages...")
            if method == "pymupdf":
                saved_images = pdf_to_images_pymupdf(pdf_path, output_folder, zoom, format)
            elif method == "pdf2image":
                saved_images = pdf_to_images_pdf2image(pdf_path, output_folder, dpi, format)
            else:
                print(f"❌ Unknown method: {method}")
                return []
        
        print(f"✅ Successfully converted {len(saved_images)} pages")
        return saved_images
        
    except Exception as e:
        print(f"❌ Error converting PDF: {e}")
        return []

def quick_convert(pdf_path, output_folder=None):
    """
    Quick conversion function with default settings
    
    Args:
        pdf_path (str): Path to PDF file
        output_folder (str): Output folder (auto-generated if None)
    
    Returns:
        list: List of saved image paths
    """
    
    if output_folder is None:
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_folder = f"{pdf_name}_images"
    
    return convert_pdf_to_images(pdf_path, output_folder, method="pymupdf", zoom=2.0)