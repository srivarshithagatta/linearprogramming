from django.shortcuts import render

from matplotlib.patches import Polygon
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import Polygon as ShapelyPolygon
import io
import base64
from scipy.spatial import ConvexHull, QhullError  # Use the Agg backend (non-GUI)


# Create your views here.
def home(request):
    return render(request, 'home.html', {'name': "raju"})

def solve(request):
    return render(request, 'solve.html')

def solve_lp(request):
    if request.method == "POST":
        method = request.POST.get("method")
        
        if method == "graphical":
            return solve_graphical(request)
        else:
            return render(request, "solve.html", {"error": "Invalid method selected."})

    return render(request, "solve.html")


def solve_graphical(request): 
    num_variables = int(request.POST.get("num_variables", 0))
    if num_variables != 2:
        return render(request, "solve.html", {"solution": "Graphical method only supports 2 variables."})

    c = [float(request.POST.get(f"obj_{i}", 0)) for i in range(num_variables)]
    
    num_constraints = int(request.POST.get("num_constraints", 0))
    
    A = []
    b = []
    for i in range(num_constraints):
        A.append([float(request.POST.get(f"constraint_{i}_{j}", 0)) for j in range(num_variables)])
        b.append(float(request.POST.get(f"rhs_{i}", 0)))

    feasible_vertices = []
    for i in range(num_constraints):
        for j in range(i + 1, num_constraints):
            A_ = np.array([A[i], A[j]])
            b_ = np.array([b[i], b[j]])
            try:
                vertex = np.linalg.solve(A_, b_)
                feasible = True
                for k in range(num_constraints):
                    if np.dot(A[k], vertex) > b[k] or (vertex < 0).any():
                        feasible = False
                        break

                if feasible:
                    feasible_vertices.append(vertex)
            except np.linalg.LinAlgError:
                continue

    if feasible_vertices:
        feasible_vertices = np.unique(feasible_vertices, axis=0)
        z_values = [np.dot(c, v) for v in feasible_vertices]
        optimal_vertex = feasible_vertices[np.argmax(z_values)]
        optimal_value = np.dot(c, optimal_vertex)  
    else:
        return render(request, "solve.html", {"solution": "No feasible solution found."})

    img = plot_constraints(A, b, feasible_vertices, optimal_vertex)

    return render(request, "solve.html", {
        "solution": f"Optimal vertex: {optimal_vertex}, Objective function value: {optimal_value}",
        "graph": img
    })

def plot_constraints(A, b, feasible_region, optimal_vertex):
    x = np.linspace(0, max(b), 400)
    plt.figure(figsize=(8, 6))

    for coeff, rhs in zip(A, b):
        if coeff[1] != 0:
            y = (rhs - coeff[0] * x) / coeff[1]
            plt.plot(x, y, label=f"{coeff[0]}x1 + {coeff[1]}x2 ≤ {rhs}")
        else:
            plt.axvline(rhs / coeff[0], color='r', linestyle='--')

    if len(feasible_region) >= 3:
        try:
            hull = ConvexHull(feasible_region)
            polygon = Polygon(feasible_region[hull.vertices], closed=True, color='lightgreen', alpha=0.5, label='Feasible Region')
            plt.gca().add_patch(polygon)

            for point in feasible_region:
                plt.plot(point[0], point[1], 'bo')  
                plt.text(point[0], point[1], f"({point[0]:.2f}, {point[1]:.2f})", fontsize=9, ha='left', color='blue')
        except Exception as e:
            print(f"Error in ConvexHull: {e}")
    else:
        print("Insufficient points to plot a feasible region.")

    if optimal_vertex is not None:
        plt.plot(optimal_vertex[0], optimal_vertex[1], 'ro', label='Optimal Solution')
        plt.text(optimal_vertex[0], optimal_vertex[1], f"({optimal_vertex[0]:.2f}, {optimal_vertex[1]:.2f})", fontsize=10, ha='left', color='red')

    plt.xlim(0, max(b) * 1.1)  
    plt.ylim(0, max(b) * 1.1)

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()

    plt.close()

    return f"data:image/png;base64,{img_str}"
