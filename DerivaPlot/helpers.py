# helpers.py

import numpy as np
import sympy as sp
from tkinter import messagebox
from scipy.integrate import quad

def validate_inputs(main_expr, x_min, x_max, order, additional_functions=None):
    """Validate all user inputs and check for empty fields."""
    # Remove whitespace
    main_expr = main_expr.strip()
    x_min = x_min.strip()
    x_max = x_max.strip()
    order = order.strip()
    
    # Check for empty fields
    if not main_expr:
        messagebox.showerror("Input Error", "Main function expression cannot be empty")
        return False, None, None, None
    
    if not x_min or not x_max or not order:
        messagebox.showerror("Input Error", "X range and derivative order cannot be empty")
        return False, None, None, None
    
    try:
        x = sp.symbols('x')
        functions = []
        
        # Predefined mathematical functions for safe evaluation
        safe_locals = {
            "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, 
            "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
            "pi": sp.pi, "e": sp.E
        }
        
        # Process main function
        sympy_expr = sp.sympify(main_expr, locals=safe_locals)
        f_main = sp.lambdify(x, sympy_expr, 'numpy')
        functions.append((main_expr, f_main))
        
        # Process additional functions if provided
        if additional_functions:
            for _, entry in additional_functions:
                expr = entry.get().strip()
                if expr:  # Only process non-empty functions
                    try:
                        sympy_expr = sp.sympify(expr, locals=safe_locals)
                        f = sp.lambdify(x, sympy_expr, 'numpy')
                        functions.append((expr, f))
                    except Exception as e:
                        raise ValueError(f"Invalid function '{expr}': {e}")
        
        # Ensure numerical validity
        x_min_val = float(x_min)
        x_max_val = float(x_max)
        order_val = int(order)
        
        if x_min_val >= x_max_val:
            raise ValueError("Min x must be less than Max x")
        
        if order_val < 1:
            raise ValueError("Derivative order must be at least 1")
            
        # Test the functions with a sample value to catch potential errors
        test_x = np.array([0.5])
        for _, f in functions:
            try:
                f(test_x)
            except Exception:
                raise ValueError("One or more functions cannot be evaluated. Check your syntax.")
        
        return True, functions, (x_min_val, x_max_val), order_val
    except Exception as e:
        messagebox.showerror("Input Error", f"Invalid input: {e}")
        return False, None, None, None

def numerical_derivative(f, x_vals, order=1):
    """Compute the numerical derivative of a function."""
    if order == 1:
        dx = x_vals[1] - x_vals[0]
        return np.gradient(f(x_vals), dx)
    else:
        # Higher order derivatives
        result = f(x_vals)
        for _ in range(order):
            dx = x_vals[1] - x_vals[0]
            result = np.gradient(result, dx)
        return result

def numerical_integral(f, x_vals):
    """Compute the numerical integral of a function."""
    return np.array([quad(f, x_vals[0], x)[0] for x in x_vals])

def update_plot_theme(fig, canvas, appearance_mode):
    """Update the plot theme to match the app theme"""
    if fig is not None:
        background_color = "#242424" if appearance_mode == "dark" else "white"
        text_color = "white" if appearance_mode == "dark" else "black"
        
        fig.patch.set_facecolor(background_color)
        axes = fig.get_axes()
        for ax in axes:
            ax.set_facecolor(background_color)
            ax.tick_params(colors=text_color)
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            ax.title.set_color(text_color)
            for spine in ax.spines.values():
                spine.set_edgecolor(text_color)
            
            # Update legend
            legend = ax.get_legend()
            if legend is not None:
                frame = legend.get_frame()
                frame.set_facecolor(background_color)
                frame.set_edgecolor(text_color)

                for text in legend.get_texts():
                    text.set_color(text_color)
        
        canvas.draw()