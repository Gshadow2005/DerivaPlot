import customtkinter as ctk
from function_visualizer import FunctionVisualizerApp

def main():
    """Main function to run the application."""
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    app = FunctionVisualizerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
