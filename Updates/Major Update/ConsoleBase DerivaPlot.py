import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import quad
import os
import sys
import time
from scipy.optimize import brentq
from pathlib import Path


class ConsoleDerivationPlotter:
    def __init__(self):
        self.functions = []
        self.x_range = (-10, 10)
        self.derivative_order = 1
        self.current_data = None
        self.fig = None
        self.default_save_dir = str(Path.home() / "Downloads")
        
    def clear_console(self):
        """Clear the console screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def print_header(self):
        """Print the application header."""
        print("\n" + "=" * 60)
        print("                      DERIVAPLOT")
        print("          Function Plotter & Derivative Visualizer")
        print("=" * 60)
        
    def print_menu(self):
        """Print the main menu options."""
        print("\nMain Menu:")
        print("1. Add/Edit Functions")
        print("2. Set X Range")
        print("3. Set Derivative Order")
        print("4. Plot Functions")
        print("5. Show Critical Values")
        print("6. Find Roots")
        print("7. Save Plot")
        print("8. Generate Function Report")
        print("9. Show Statistics")
        print("0. Exit")
        print("-" * 60)
        
    def validate_function(self, function_expr):
        """Validate a function expression."""
        try:
            x = sp.symbols('x')
            sympy_expr = sp.sympify(function_expr, locals={
                "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, 
                "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt, 
                "pi": sp.pi, "e": sp.E
            })
            f = sp.lambdify(x, sympy_expr, 'numpy')

            test_x = np.array([0.5])
            f(test_x)
            
            return True, f, function_expr
        except Exception as e:
            print(f"Error: Invalid function '{function_expr}': {e}")
            return False, None, function_expr
    
    def manage_functions(self):
        """Add or edit functions."""
        self.clear_console()
        self.print_header()
        
        print("\nCurrent Functions:")
        if not self.functions:
            print("  No functions added yet.")
        else:
            for i, (expr, _) in enumerate(self.functions):
                print(f"  {i+1}. {expr}")
        
        print("\nFunction Management:")
        print("1. Add New Function")
        print("2. Edit Existing Function")
        print("3. Remove Function")
        print("4. Return to Main Menu")
        
        choice = input("\nEnter choice (1-4): ")
        
        if choice == '1':
            function_expr = input("Enter function expression (use 'x' as variable, e.g., sin(x) + x**2): ")
            valid, f, expr = self.validate_function(function_expr)
            if valid:
                self.functions.append((expr, f))
                print(f"Function '{expr}' added successfully.")
        
        elif choice == '2' and self.functions:
            try:
                idx = int(input(f"Enter function number to edit (1-{len(self.functions)}): ")) - 1
                if 0 <= idx < len(self.functions):
                    function_expr = input(f"Enter new expression for function {idx+1} (current: {self.functions[idx][0]}): ")
                    valid, f, expr = self.validate_function(function_expr)
                    if valid:
                        self.functions[idx] = (expr, f)
                        print(f"Function {idx+1} updated successfully.")
                else:
                    print("Invalid function number.")
            except ValueError:
                print("Please enter a valid number.")
        
        elif choice == '3' and self.functions:
            try:
                idx = int(input(f"Enter function number to remove (1-{len(self.functions)}): ")) - 1
                if 0 <= idx < len(self.functions):
                    removed = self.functions.pop(idx)
                    print(f"Function '{removed[0]}' removed successfully.")
                else:
                    print("Invalid function number.")
            except ValueError:
                print("Please enter a valid number.")
        
        input("\nPress Enter to continue...")
    
    def set_x_range(self):
        """Set the x-axis range for plotting."""
        self.clear_console()
        self.print_header()
        
        print(f"\nCurrent X Range: {self.x_range}")
        try:
            x_min = float(input("Enter minimum x value: "))
            x_max = float(input("Enter maximum x value: "))
            
            if x_min >= x_max:
                print("Error: Minimum x must be less than maximum x.")
            else:
                self.x_range = (x_min, x_max)
                print(f"X Range updated to {self.x_range}")
        except ValueError:
            print("Error: Please enter valid numbers.")
        
        input("\nPress Enter to continue...")
    
    def set_derivative_order(self):
        """Set the derivative order."""
        self.clear_console()
        self.print_header()
        
        print(f"\nCurrent Derivative Order: {self.derivative_order}")
        try:
            order = int(input("Enter derivative order (1 or higher): "))
            if order < 1:
                print("Error: Derivative order must be at least 1.")
            else:
                self.derivative_order = order
                print(f"Derivative order updated to {self.derivative_order}")
        except ValueError:
            print("Error: Please enter a valid integer.")
        
        input("\nPress Enter to continue...")
    
    def numerical_derivative(self, f, x_vals, order=1):
        """Calculate numerical derivative of a function."""
        if order == 1:
            dx = x_vals[1] - x_vals[0]
            return np.gradient(f(x_vals), dx)
        else:
            result = f(x_vals)
            for _ in range(order):
                dx = x_vals[1] - x_vals[0]
                result = np.gradient(result, dx)
            return result

    def numerical_integral(self, f, x_vals):
        """Calculate numerical integral of a function."""
        return np.array([quad(f, x_vals[0], x)[0] for x in x_vals])
    
    def plot_functions(self):
        """Plot the functions, their derivatives, and integrals."""
        self.clear_console()
        self.print_header()
        
        if not self.functions:
            print("\nError: No functions to plot. Please add at least one function.")
            input("\nPress Enter to continue...")
            return
        
        try:
            print("\nPlotting functions...")

            if self.fig is not None:
                plt.close(self.fig)
            
            x_vals = np.linspace(self.x_range[0], self.x_range[1], 400)
            self.fig, ax = plt.subplots(figsize=(10, 6))
            
            colors = plt.cm.tab10.colors
            all_functions_data = []
            
            for i, (expr, f) in enumerate(self.functions):
                color_idx = i % len(colors)
                base_color = colors[color_idx]
                
                y_vals = f(x_vals)
                dydx_vals = self.numerical_derivative(f, x_vals, self.derivative_order)
                integral_vals = self.numerical_integral(f, x_vals)
                
                ax.plot(x_vals, y_vals, label=f'Function: {expr}', color=base_color, linewidth=2)
                ax.plot(x_vals, dydx_vals, label=f'{self.derivative_order}-Order Derivative of {expr}', 
                    color=base_color, linestyle='dashed', linewidth=1.5)
                ax.plot(x_vals, integral_vals, label=f'Integral of {expr}', 
                    color=base_color, linestyle='dotted', linewidth=1.5)
                
                all_functions_data.append({
                    "expr": expr, 
                    "function": f, 
                    "y_vals": y_vals, 
                    "derivative": dydx_vals, 
                    "integral": integral_vals
                })
            
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Functions, Derivatives, and Integrals')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            self.current_data = {
                "functions": [{"expr": expr} for expr, _ in self.functions], 
                "x_range": self.x_range, 
                "order": self.derivative_order,
                "function_data": all_functions_data
            }

            plt.show(block=False)
            print("\nPlot created successfully.")
            print("(Close the plot window to continue)")
            plt.waitforbuttonpress()
            
        except Exception as e:
            print(f"\nError plotting functions: {e}")
            input("\nPress Enter to continue...")
    
    def find_critical_values(self, function_expr, x_range):
        """Find critical values of a function."""
        try:
            x = sp.Symbol('x')
            expr = sp.sympify(function_expr, locals={
                "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, 
                "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt, 
                "pi": sp.pi, "e": sp.E
            })
            derivative = sp.diff(expr, x)

            critical_points = sp.nroots(derivative)
            valid_critical_points = [
                point for point in critical_points 
                if x_range[0] <= float(point.evalf()) <= x_range[1] and sp.im(point) == 0
            ]
            
            if not valid_critical_points:
                return []
            
            critical_values = []
            for point in valid_critical_points:
                point_val = float(point.evalf())
                func_val = float(expr.subs(x, point))
                derivative_val = float(derivative.subs(x, point))
                critical_values.append({
                    'x': point_val, 
                    'y': func_val, 
                    'derivative': derivative_val
                })
            
            return critical_values
        
        except Exception as e:
            print(f"Error finding critical values: {e}")
            return []
    
    def show_critical_values(self):
        """Display critical values and plot them."""
        self.clear_console()
        self.print_header()
        
        if not self.functions:
            print("\nError: No functions available. Please add at least one function.")
            input("\nPress Enter to continue...")
            return
        
        try:
            print("\nCalculating critical values...")

            if self.fig is not None:
                plt.close(self.fig)
            
            x_vals = np.linspace(self.x_range[0], self.x_range[1], 400)
            self.fig, ax = plt.subplots(figsize=(10, 6))
            
            colors = plt.cm.tab10.colors
            critical_values_data = []

            print("\nCritical Values:")
            print("-" * 60)
            
            for i, (expr, f) in enumerate(self.functions):
                color_idx = i % len(colors)
                base_color = colors[color_idx]
                
                y_vals = f(x_vals)
                critical_values = self.find_critical_values(expr, self.x_range)
                ax.plot(x_vals, y_vals, label=f'Function: {expr}', color=base_color, linewidth=2)

                print(f"\nFunction: {expr}")
                if critical_values:
                    cv_x = [point['x'] for point in critical_values]
                    cv_y = [point['y'] for point in critical_values]
                    
                    ax.scatter(cv_x, cv_y, color='red', s=100, zorder=5, label=f'Critical Points of {expr}')
                    
                    for j, point in enumerate(critical_values):
                        print(f"  Critical Point {j+1}: x = {point['x']:.4f}, y = {point['y']:.4f}, derivative = {point['derivative']:.4f}")
                        
                        ax.annotate(
                            f"x={point['x']:.2f}\ny={point['y']:.2f}", 
                            (point['x'], point['y']), 
                            xytext=(10, 10), 
                            textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                        )
                else:
                    print("  No critical values found within the specified range.")
                
                critical_values_data.append({
                    "expr": expr, 
                    "critical_values": critical_values
                })
            
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Functions with Critical Values')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            self.current_data = {
                "functions": [{"expr": expr} for expr, _ in self.functions], 
                "x_range": self.x_range, 
                "critical_values": critical_values_data
            }

            plt.show(block=False)
            print("\nPlot with critical values created successfully.")
            print("(Close the plot window to continue)")

            plt.waitforbuttonpress()
            
        except Exception as e:
            print(f"\nError calculating critical values: {e}")
            
        input("\nPress Enter to continue...")
    
    def find_roots(self, function_expr):
        """Find roots of a function."""
        try:
            x = sp.Symbol('x')
            expr = sp.sympify(function_expr, locals={
                "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, 
                "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt, 
                "pi": sp.pi, "e": sp.E
            })
            

            try:
                symbolic_roots = sp.solve(expr, x)
                roots = [float(root.evalf()) for root in symbolic_roots if root.is_real]
            except Exception:
                roots = []

                sp.lambdify(x, expr, 'numpy')
                
                def f(val):
                    return float(expr.subs(x, val))

                intervals = [(-100, -10), (-10, -1), (-1, 0), (0, 1), (1, 10), (10, 100)]
                
                for a, b in intervals:
                    try:
                        if f(a) * f(b) <= 0: 
                            root = brentq(f, a, b)
                            roots.append(root)
                    except Exception:
                        continue

            valid_roots = []
            for root in roots:
                try:
                    root_val = float(root)
                    if not any(abs(root_val - existing) < 1e-10 for existing in valid_roots):
                        valid_roots.append(root_val)
                except Exception:
                    pass
                    
            return valid_roots
            
        except Exception as e:
            print(f"Error finding roots: {e}")
            return []
    
    def show_roots(self):
        """Display roots for each function."""
        self.clear_console()
        self.print_header()
        
        if not self.functions:
            print("\nError: No functions available. Please add at least one function.")
            input("\nPress Enter to continue...")
            return
        
        print("\nCalculating function roots...\n")
        print("Function Roots:")
        print("-" * 60)
        
        for expr, _ in self.functions:
            print(f"\nFunction: {expr}")
            roots = self.find_roots(expr)
            
            if roots:
                for i, root in enumerate(roots):
                    if abs(root - round(root)) < 1e-10:
                        print(f"  Root {i+1}: x = {int(round(root))}")
                    else:
                        print(f"  Root {i+1}: x = {root:.6f}")
            else:
                print("  No real roots found")
        
        input("\nPress Enter to continue...")
    
    def get_save_directory(self):
        """Get directory path from user for saving files."""
        print("\nSave Location:")
        print(f"Default: {self.default_save_dir}")
        print("\nOptions:")
        print("1. Use default save location")
        print("2. Specify custom directory")
        
        choice = input("\nEnter choice (1-2): ")
        
        if choice == '2':
            while True:
                dir_path = input("\nEnter directory path (or 'cd' to browse): ")
                
                if dir_path.lower() == 'cd':
                    self.clear_console()
                    self.print_header()
                    print("\nBrowse Directories:")
                    
                    current_dir = os.getcwd()
                    print(f"Current directory: {current_dir}")
                    
                    while True:
                        try:
                            dirs = [d for d in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, d))]
                            
                            print("\nAvailable directories:")
                            for i, d in enumerate(dirs):
                                print(f"{i+1}. {d}")
                            print(f"{len(dirs)+1}. Parent directory (..)") 
                            print(f"{len(dirs)+2}. Select current directory")
                            
                            dir_choice = input("\nEnter choice number or directory name: ")
                            
                            if dir_choice.isdigit():
                                choice_num = int(dir_choice)
                                if 1 <= choice_num <= len(dirs):
                                    current_dir = os.path.join(current_dir, dirs[choice_num-1])
                                elif choice_num == len(dirs) + 1:
                                    current_dir = os.path.dirname(current_dir)
                                elif choice_num == len(dirs) + 2:
                                    return current_dir
                            else:
                                # User typed a directory name
                                new_dir = os.path.join(current_dir, dir_choice)
                                if os.path.isdir(new_dir):
                                    current_dir = new_dir
                                else:
                                    print(f"Directory '{dir_choice}' not found.")
                            
                            self.clear_console()
                            self.print_header()
                            print("\nBrowse Directories:")
                            print(f"Current directory: {current_dir}")
                            
                        except Exception as e:
                            print(f"Error: {e}")
                            input("\nPress Enter to continue...")
                else:
                    if os.path.isdir(dir_path):
                        return dir_path
                    else:
                        print(f"Error: Directory '{dir_path}' not found or is not accessible.")
                        if input("Try again? (y/n): ").lower() != 'y':
                            return self.default_save_dir
        
        return self.default_save_dir
    
    def save_plot(self):
        """Save the current plot as an image file."""
        self.clear_console()
        self.print_header()
        
        if self.fig is None:
            print("\nError: No plot to save. Please create a plot first.")
            input("\nPress Enter to continue...")
            return
        
        print("\nSave Plot:")
        filename = input("Enter filename (without extension): ")
        
        if not filename:
            print("Save cancelled.")
            input("\nPress Enter to continue...")
            return
        
        # Get save directory from user
        save_dir = self.get_save_directory()
        
        print("\nSelect file format:")
        print("1. PNG")
        print("2. JPEG")
        print("3. PDF")
        print("4. SVG")
        
        choice = input("\nEnter choice (1-4): ")
        
        extension_map = {"1": "png", "2": "jpg", "3": "pdf", "4": "svg"}
        if choice in extension_map:
            extension = extension_map[choice]
            filepath = os.path.join(save_dir, f"{filename}.{extension}")
            
            try:
                self.fig.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"\nPlot saved successfully as '{filepath}'")
            except Exception as e:
                print(f"\nError saving plot: {e}")
        else:
            print("\nInvalid choice. Save cancelled.")
        
        input("\nPress Enter to continue...")
    
    def calculate_statistics(self):
        """Calculate statistics for the plotted functions."""
        if not self.functions:
            return None
        
        try:
            x_vals = np.linspace(self.x_range[0], self.x_range[1], 400)
            all_y_vals = []
            
            for _, f in self.functions:
                y_vals = f(x_vals)
                all_y_vals.append(y_vals)
            
            all_y_vals = np.concatenate(all_y_vals)
            
            # Calculate areas under curves
            areas = []
            for _, f in self.functions:
                y_vals = f(x_vals)
                area = np.trapezoid(y_vals, x_vals)
                areas.append(area)
            
            stats = {
                "max_value": np.max(all_y_vals),
                "min_value": np.min(all_y_vals),
                "mean_value": np.mean(all_y_vals),
                "std_deviation": np.std(all_y_vals),
                "areas": areas,
                "total_area": np.sum(areas)
            }
            
            return stats
        
        except Exception as e:
            print(f"Error calculating statistics: {e}")
            return None
    
    def show_statistics(self):
        """Display statistics for the current functions."""
        self.clear_console()
        self.print_header()
        
        if not self.functions:
            print("\nError: No functions available. Please add at least one function.")
            input("\nPress Enter to continue...")
            return
        
        print("\nCalculating statistics...\n")
        
        stats = self.calculate_statistics()
        
        if stats:
            print("Function Statistics:")
            print("-" * 60)
            print(f"Maximum Value: {stats['max_value']:.4f}")
            print(f"Minimum Value: {stats['min_value']:.4f}")
            print(f"Mean Value: {stats['mean_value']:.4f}")
            print(f"Standard Deviation: {stats['std_deviation']:.4f}")
            
            print("\nArea Under Curve:")
            for i, (expr, _) in enumerate(self.functions):
                print(f"  {expr}: {stats['areas'][i]:.4f}")
            
            print(f"Total Area: {stats['total_area']:.4f}")
        else:
            print("Could not calculate statistics.")
        
        input("\nPress Enter to continue...")
    
    def generate_report(self):
        """Generate a text report for the current functions."""
        self.clear_console()
        self.print_header()
        
        if not self.functions:
            print("\nError: No functions available. Please add at least one function.")
            input("\nPress Enter to continue...")
            return
        
        if self.fig is None:
            print("\nError: No plot data available. Please create a plot first.")
            input("\nPress Enter to continue...")
            return
        
        print("\nGenerating function report...\n")
        
        filename = input("Enter filename for the report (without extension): ")
        
        if not filename:
            print("Report generation cancelled.")
            input("\nPress Enter to continue...")
            return
        
        # Get save directory from user
        save_dir = self.get_save_directory()
        
        filepath = os.path.join(save_dir, f"{filename}.txt")
        
        try:
            with open(filepath, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("             DERIVAPLOT FUNCTION ANALYSIS REPORT\n")
                f.write("=" * 60 + "\n\n")
                
                f.write("FUNCTIONS:\n")
                for i, (expr, _) in enumerate(self.functions):
                    f.write(f"  Function {i+1}: {expr}\n")
                
                f.write(f"\nX RANGE: [{self.x_range[0]}, {self.x_range[1]}]\n")
                f.write(f"DERIVATIVE ORDER: {self.derivative_order}\n\n")

                if self.current_data and 'critical_values' in self.current_data:
                    f.write("CRITICAL VALUES:\n")
                    for func_data in self.current_data['critical_values']:
                        f.write(f"  Function: {func_data['expr']}\n")
                        if func_data['critical_values']:
                            for i, cv in enumerate(func_data['critical_values']):
                                f.write(f"    Point {i+1}: x = {cv['x']:.4f}, y = {cv['y']:.4f}\n")
                        else:
                            f.write("    No critical values found\n")
                    f.write("\n")

                f.write("ROOTS:\n")
                for expr, _ in self.functions:
                    f.write(f"  Function: {expr}\n")
                    roots = self.find_roots(expr)
                    if roots:
                        for i, root in enumerate(roots):
                            if abs(root - round(root)) < 1e-10:
                                f.write(f"    Root {i+1}: x = {int(round(root))}\n")
                            else:
                                f.write(f"    Root {i+1}: x = {root:.6f}\n")
                    else:
                        f.write("    No real roots found\n")
                f.write("\n")

                stats = self.calculate_statistics()
                if stats:
                    f.write("STATISTICS:\n")
                    f.write(f"  Maximum Value: {stats['max_value']:.4f}\n")
                    f.write(f"  Minimum Value: {stats['min_value']:.4f}\n")
                    f.write(f"  Mean Value: {stats['mean_value']:.4f}\n")
                    f.write(f"  Standard Deviation: {stats['std_deviation']:.4f}\n\n")
                    
                    f.write("  Area Under Curve:\n")
                    for i, (expr, _) in enumerate(self.functions):
                        f.write(f"    {expr}: {stats['areas'][i]:.4f}\n")
                    f.write(f"  Total Area: {stats['total_area']:.4f}\n\n")

                from datetime import datetime
                f.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("\nThank you for using DerivaPlot")
            
            print(f"\nReport saved successfully as '{filepath}'")

            plot_filepath = os.path.join(save_dir, f"{filename}_plot.png")
            self.fig.savefig(plot_filepath, dpi=300, bbox_inches='tight')
            print(f"Plot image saved as '{plot_filepath}'")
            
        except Exception as e:
            print(f"\nError generating report: {e}")
        
        input("\nPress Enter to continue...")
    
    def run(self):
        """Run the main application loop."""
        while True:
            self.clear_console()
            self.print_header()
            self.print_menu()
            
            choice = input("Enter your choice (0-9): ")
            
            if choice == '0':
                print("\nExiting DerivaPlot. Goodbye!")
                time.sleep(2)
                break
            elif choice == '1':
                self.manage_functions()
            elif choice == '2':
                self.set_x_range()
            elif choice == '3':
                self.set_derivative_order()
            elif choice == '4':
                self.plot_functions()
            elif choice == '5':
                self.show_critical_values()
            elif choice == '6':
                self.show_roots()
            elif choice == '7':
                self.save_plot()
            elif choice == '8':
                self.generate_report()
            elif choice == '9':
                self.show_statistics()
            else:
                print("\nInvalid choice. Please enter a number between 0 and 9.")
                input("\nPress Enter to continue...")


def main():
    """Main entry point for the application."""
    try:
        app = ConsoleDerivationPlotter()
        app.run()
    except KeyboardInterrupt:
        print("\n\nProgram terminated by user. Goodbye!")
        time.sleep(2)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        input("\nPress Enter to exit...")
        time.sleep(2) 


if __name__ == "__main__":
    main()