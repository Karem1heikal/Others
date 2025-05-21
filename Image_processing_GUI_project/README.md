# Digital Image Processing GUI – PyQt5

This project is a GUI-based application developed using PyQt5 as part of a Digital Image Processing course. It integrates all the core concepts and techniques covered in the course into an interactive, user-friendly interface.

## Features

- Grayscale Conversion  
- Histogram Visualization  
- Histogram Stretching  
- Histogram Equalization  
- Histogram Sliding  
- Multi-tab Interface to organize different image processing functions
- Real-time Image Display and Processing (no need for image paths – processed in-memory)

## Technologies Used

- Python 3
- PyQt5
- NumPy
- Matplotlib
- Pillow (PIL)

## Project Structure

DigitalImageProcessingGUI/ ├── main.py               
# Entry point of the application ├── gui/                  
# Contains all PyQt5 GUI components │   ├── tab1.py           
# Grayscale + Histogram │   ├── tab2.py           
# Histogram Stretching │   ├── tab3.py           
# Histogram Sliding │   └── tab4.py           
# Histogram Equalization ├── assets/               
# Sample images used in the GUI (optional) └── README.md             # Project documentation

## How to Run

1. Clone the repository:
`bash
git clone https://github.com/Karem1heikal/Others/tree/main/Image_processing_GUI_project
cd DigitalImageProcessingGUI

python main.py