from arduino import initialize_arduino
from app_ui import demo

if __name__ == "__main__":
    initialize_arduino()
    demo.launch()
