import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.misc import derivative
from scipy.integrate import quad
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw, ImageFont

# Set customtkinter appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

def numerical_derivative(f, x_vals, order=1):
    """Compute the numerical derivative of a function."""
    return np.array([derivative(f, x, dx=1e-5, n=order) for x in x_vals])

def numerical_integral(f, x_vals):
    """Compute the numerical integral of a function."""
    return np.array([quad(f, 0, x)[0] for x in x_vals])

def plot_functions(f, f_expr, x_range, derivative_order):
    """Plot the function, its derivative, and its integral."""
    x_vals = np.linspace(x_range[0], x_range[1], 400)
    y_vals = f(x_vals)
    dydx_vals = numerical_derivative(f, x_vals, derivative_order)
    integral_vals = numerical_integral(f, x_vals)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label=f'Function: {f_expr}')
    plt.plot(x_vals, dydx_vals, label=f'{derivative_order}-Order Derivative', linestyle='dashed')
    plt.plot(x_vals, integral_vals, label='Integral', linestyle='dotted')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.title('Function, Derivative, and Integral')
    
    # Save the graph temporarily
    graph_path = "temp_graph.png"
    plt.savefig(graph_path)
    plt.show()
    return graph_path

def save_receipt(f_expr, x_range, derivative_order, graph_path):
    """Save a receipt with function details and the graph."""
    receipt_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
    if not receipt_path:
        return
    
    # Create receipt image
    receipt_width, receipt_height = 500, 600
    image = Image.new('RGB', (receipt_width, receipt_height), 'white')
    draw = ImageDraw.Draw(image)
    
    font = ImageFont.load_default()
    draw.text((20, 20), "Function Visualizer Receipt", fill="black", font=font)
    draw.text((20, 60), f"Function: {f_expr}", fill="black", font=font)
    draw.text((20, 100), f"X Range: {x_range}", fill="black", font=font)
    draw.text((20, 140), f"Derivative Order: {derivative_order}", fill="black", font=font)
    
    # Add graph to receipt
    graph_img = Image.open(graph_path)
    graph_img = graph_img.resize((450, 300))
    image.paste(graph_img, (25, 180))
    
    # Save receipt
    image.save(receipt_path)
    messagebox.showinfo("Success", f"Receipt saved as {receipt_path}")

def validate_inputs(expr, x_min, x_max, order):
    """Validate user inputs and properly convert functions."""
    try:
        x = sp.symbols('x')
        # Convert the expression into a sympy function
        sympy_expr = sp.sympify(expr, locals={"sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt})
        # Convert to a lambda function that NumPy can evaluate
        f = sp.lambdify(x, sympy_expr, 'numpy')
        
        # Ensure numerical validity
        float(x_min)
        float(x_max)
        int(order)
        
        return True, f
    except Exception as e:
        messagebox.showerror("Input Error", f"Invalid input: {e}")
        return False, None


def main():
    """Main function to run the application."""
    root = ctk.CTk()
    root.geometry("400x400")
    root.title("Function Visualizer")
    
    def on_submit():
        """Handle the submit button click."""
        expr = entry_func.get()
        x_min = entry_xmin.get()
        x_max = entry_xmax.get()
        order = entry_order.get()
        
        # Validate inputs
        is_valid, f = validate_inputs(expr, x_min, x_max, order)
        if not is_valid:
            return
        
        # Plot functions and save graph
        graph_path = plot_functions(f, expr, (float(x_min), float(x_max)), int(order))
        
        # Save receipt
        save_receipt(expr, (x_min, x_max), order, graph_path)
    
    # UI Elements
    label_func = ctk.CTkLabel(root, text="Enter function (e.g., x**2 + 3*x + 5):")
    label_func.pack(pady=10)
    entry_func = ctk.CTkEntry(root, width=300)
    entry_func.pack(pady=5)
    
    label_xmin = ctk.CTkLabel(root, text="Min x value:")
    label_xmin.pack(pady=5)
    entry_xmin = ctk.CTkEntry(root, width=100)
    entry_xmin.pack(pady=5)
    
    label_xmax = ctk.CTkLabel(root, text="Max x value:")
    label_xmax.pack(pady=5)
    entry_xmax = ctk.CTkEntry(root, width=100)
    entry_xmax.pack(pady=5)
    
    label_order = ctk.CTkLabel(root, text="Derivative order:")
    label_order.pack(pady=5)
    entry_order = ctk.CTkEntry(root, width=100)
    entry_order.pack(pady=5)
    
    btn_submit = ctk.CTkButton(root, text="Plot and Save Receipt", command=on_submit)
    btn_submit.pack(pady=20)
    
    root.mainloop()

if __name__ == "__main__":
    main()



