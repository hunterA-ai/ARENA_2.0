# %%
import os
import sys
import torch as t
from torch import Tensor
import einops
from ipywidgets import interact
import plotly.express as px
from ipywidgets import interact
from pathlib import Path
from IPython.display import display
from jaxtyping import Float, Int, Bool, Shaped, jaxtyped
import typeguard

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part1_ray_tracing"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow
from part1_ray_tracing.utils import render_lines_with_plotly, setup_widget_fig_ray, setup_widget_fig_triangle
import part1_ray_tracing.tests as tests

MAIN = __name__ == "__main__"

# %%
def make_rays_1d(num_pixels: int, y_limit: float) -> t.Tensor:
    '''
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) where the num_points dimension contains (origin, direction) and the num_dim dimension contains xyz.

    Example of make_rays_1d(9, 1.0): [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    '''
    # Assume num_pixels != 0
    step_size = (2 * y_limit) / (num_pixels - 1)
    y_points = t.arange(-y_limit, y_limit + step_size, step_size)
    array_tensor = [ [[0,0,0], [1, y_points[i], 0]] for i in range(len(y_points))]
    tensor = t.tensor(array_tensor)

    rays = t.zeros((num_pixels, 2, 3), dtype=t.float32)
    t.linspace(-y_limit, y_limit, num_pixels, out=rays[:, 1, 1])
    rays[:, 0, 0] = 1
    return tensor

rays1d = make_rays_1d(9, 10.0)

if MAIN:
    fig = render_lines_with_plotly(rays1d)
# %%
if MAIN:
    fig = setup_widget_fig_ray()
    display(fig)

@interact
def response(seed=(0, 10, 1), v=(-2.0, 2.0, 0.01)):
    t.manual_seed(seed)
    L_1, L_2 = t.rand(2, 2)
    P = lambda v: L_1 + v * (L_2 - L_1)
    x, y = zip(P(-2), P(2))
    with fig.batch_update(): 
        fig.data[0].update({"x": x, "y": y}) 
        fig.data[1].update({"x": [L_1[0], L_2[0]], "y": [L_1[1], L_2[1]]}) 
        fig.data[2].update({"x": [P(v)[0]], "y": [P(v)[1]]})
# %%
segments = t.tensor([
    [[1.0, -12.0, 0.0], [1, -6.0, 0.0]], 
    [[0.5, 0.1, 0.0], [0.5, 1.15, 0.0]], 
    [[2, 12.0, 0.0], [2, 21.0, 0.0]]
])

render_lines_with_plotly(segments, 2 *rays1d)
# %%
def intersect_ray_1d(ray: t.Tensor, segment: t.Tensor) -> bool:
    '''
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    '''
    # Du + (L2-L1)v = L_1 - O
    # (Du + (L2-L1)v)_x
    # (Du + (L2-L1)v)_y

    # (D + (L2-L1))_x * (u, v) = (L_1 - O)_x
    # (D + (L2-L1))_y * (u, v) = (L_1 - O)_y
    ray = ray[..., :2]
    segment = segment[..., :2]
    O, D = ray
    L_1, L_2 = segment
    L_12 = L_1 - L_2
    A = t.stack((D, L_12), dim=-1)
    B = L_1 - O
    try:
        sol = t.linalg.solve(A, B)
    except RuntimeError:
        return False
    
    u = sol[0].item()
    v = sol[1].item()
    return (u > 0.0) & (v > 0.0) & (v < 1.0)


if MAIN:
    tests.test_intersect_ray_1d(intersect_ray_1d)
    tests.test_intersect_ray_1d_special_case(intersect_ray_1d)
# %%
x = t.randn(2, 3)
x_repeated = einops.repeat(x, 'a b -> a b c', c=4)
print(x_repeated)
# %%
def intersect_rays_1d(rays: Float[Tensor, "nrays 2 3"], segments: Float[Tensor, "nsegments 2 3"]) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, return True if it intersects any segment.
    '''
    # For each 
    # (D , (L2-L1))_x * (u, v) = (L_1 - O)_x
    # (D , (L2-L1))_y * (u, v) = (L_1 - O)_y
    # then change L1 and L2
    # (D , (L2-L1))_x * (u, v) = (L_1 - O)_x
    # (D , (L2-L1))_y * (u, v) = (L_1 - O)_y

    # B = L_1 - O can be (nsegments 2)
    # A = t.stack((D, L_12), dim=-1) can be (nsegments 2 2)
    # This is for one batch, now lets do it for all batches
    # B = L_1 - 0 can be ([nrays * nsegments] 2)
    # A = t.stack((D, L_12), dim=-1) can be ([nrays * nsegments] 2 2)
    # Then we can filter u>0 & v>0 & v<1 for all numbers
    # Then we can aggregate each nsegments terms
    rays = rays[..., :2]
    segments = segments[..., :2]
    nsegments = segments.shape[0]
    nrays = rays.shape[0]
    # New (vectorized) solution
    big_rays = einops.repeat(rays, "r c d -> r repeat c d", repeat=nsegments)
    big_segments = einops.repeat(segments, "s c d -> repeat s c d", repeat=nrays)
    O = big_rays[..., 0, :]
    D = big_rays[..., 1, :]
    L1 = big_segments[..., 0, :]
    L2 = big_segments[..., 1, :]
    L12 = L1 - L2
    B = L1 - O
    A = t.stack((D,L12), dim=-1)
    invertible_filter = t.linalg.det(A).abs() < 1e-6
    A[invertible_filter] = t.eye(n=2)
    sol = t.linalg.solve(A, B) # (nrays nsegments 2)
    bool_tens = (sol[..., 0] >= 0) & (sol[..., 1] <= 1) & (sol[..., 1] >= 0) & ~(invertible_filter)
    return t.any(bool_tens, dim=1)


    # Old solution
    bool_tensor = t.zeros(nrays).bool()
    for i in range(nrays):
        big_rays = einops.repeat(rays[i], "c d -> repeat c d", repeat=nsegments)
        O = big_rays[:, 0, :]
        D = big_rays[:, 1, :]
        L1 = segments[:, 0, :]
        L2 = segments[:, 1, :]
        L12 = L1 - L2
        B = L1 - O
        A = t.stack((D,L12), dim=-1)
        try:
            sol = t.linalg.solve(A, B) # ([nsegments] 2)
            bool_tens = (sol[:, 0] >= 0) & (sol[:, 1] <= 1) & (sol[:, 1] >= 0)
            bool_tensor[i] = t.any(bool_tens).item()
        except:
            bool_tensor[i] = False
    return bool_tensor
        
            




if MAIN:
    tests.test_intersect_rays_1d(intersect_rays_1d)
    tests.test_intersect_rays_1d_special_case(intersect_rays_1d)
# %%
def make_rays_2d(num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float) -> Float[t.Tensor, "nrays 2 3"]:
    '''
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    '''
    rays = t.zeros((num_pixels_y * num_pixels_z, 2, 3))
    t.ones((num_pixels_y * num_pixels_z), out=rays[:, 1, 0])
    # rays[:,1,0] = 1
    rays[:,1,1]=einops.repeat(t.linspace(-y_limit, y_limit, num_pixels_y), "a -> (repeat a)", repeat=num_pixels_z)
    rays[:,1,2]=einops.repeat(t.linspace(-z_limit, z_limit, num_pixels_z), "a -> (a repeat)", repeat=num_pixels_y)
    return rays

if MAIN:
    rays_2d = make_rays_2d(10, 10, 0.3, 0.3)
    render_lines_with_plotly(rays_2d)
# %%
if MAIN:
    one_triangle = t.tensor([[0, 0, 0], [3, 0.5, 0], [2, 3, 0]])
    A, B, C = one_triangle
    x, y, z = one_triangle.T

    fig = setup_widget_fig_triangle(x, y, z)

@interact(u=(-0.5, 1.5, 0.01), v=(-0.5, 1.5, 0.01))
def response(u=0.0, v=0.0):
    P = A + u * (B - A) + v * (C - A)
    fig.data[2].update({"x": [P[0]], "y": [P[1]]})


if MAIN:
    display(fig)
# %%
Point = Float[Tensor, "points=3"]

@jaxtyped
@typeguard.typechecked
def triangle_ray_intersects(A: Point, B: Point, C: Point, O: Point, D: Point) -> bool:
    '''
    A: shape (3,), one vertex of the triangle
    B: shape (3,), second vertex of the triangle
    C: shape (3,), third vertex of the triangle
    O: shape (3,), origin point
    D: shape (3,), direction point

    Return True if the ray and the triangle intersect.
    '''
    # Solving for (u,v,s) in A + u(B-A) + v(C-A) = O + sD
    # { [(B-A) (C-A) (-D)] [u v s] }_x = { [O - A] }_x
    # Where A,B,C,D,O are all vectors in three dimensions, we will stack the matricies along their last component
    # to get component-wise operations (B-A)_x * u + (C-A)_x * v + (-D)_x * s = (O-A)_x
    x1 = (B-A)
    x2 = (C-A)
    x3 = (-D)
    b = (O-A)
    equation_matrix = t.stack((x1,x2,x3), dim=-1)
    solution_vector = b
    try:
        sols = t.linalg.solve(equation_matrix, solution_vector)
        return (t.all(sols >= 0) & ((sols[0] + sols[1]) <= 1)).item()
    except:
        return False
if MAIN:
    tests.test_triangle_ray_intersects(triangle_ray_intersects)
# %%
def raytrace_triangle(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"]
) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, return True if the triangle intersects that ray.
    '''
    # Solving for (u,v,s) in A + u(B-A) + v(C-A) = O + sD
    # { [(B-A) (C-A) (-D)] [u v s] }_x = { [O - A] }_x
    # Where A,B,C,D,O are all vectors in three dimensions, we will stack the matricies along their last component
    # to get component-wise operations (B-A)_x * u + (C-A)_x * v + (-D)_x * s = (O-A)_x

    nrays = rays.shape[0]
    big_triangle = einops.repeat(triangle, "p d -> repeat p d", repeat=nrays)
    A, B, C = big_triangle.unbind(dim=1) # shape(batch, dims) for each triangle coordinate
    O, D = rays.unbind(dim=1) # shape(batch, dims) for origin and vector
    x1 = (B-A)
    x2 = (C-A)
    x3 = (-D)
    batch_sol = (O-A) # shape(batch, dims)
    batch_eqs = t.stack((x1,x2,x3), dim=-1) # collect [dims, variables] matrix, so that each row is a dimension. shape(batch, dims, variables)
    zero_det_filter = t.linalg.det(batch_eqs).abs() < 1e-6
    batch_eqs[zero_det_filter] = t.eye(3)
    sol = t.linalg.solve(batch_eqs, batch_sol)
    u, v, s = sol.unbind(dim=-1) # each is size(batch_size, 1), create mask so that they are in the correct range
    return t.all(sol >= 0, dim=-1) & (u+v <= 1) & ~(zero_det_filter)


if MAIN:
    A = t.tensor([1, 0.0, -0.5])
    B = t.tensor([1, -0.5, 0.0])
    C = t.tensor([1, 0.5, 0.5])
    num_pixels_y = num_pixels_z = 20
    y_limit = z_limit = 0.5

    # Plot triangle & rays
    test_triangle = t.stack([A, B, C], dim=0)
    rays2d = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
    triangle_lines = t.stack([A, B, C, A, B, C], dim=0).reshape(-1, 2, 3)
    render_lines_with_plotly(rays2d, triangle_lines)

    # Calculate and display intersections
    intersects = raytrace_triangle(rays2d, test_triangle)
    img = intersects.reshape(num_pixels_y, num_pixels_z).int()
    imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")

# %%
# Incorrect Function
def raytrace_triangle_with_bug(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"]
) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, return True if the triangle intersects that ray.
    '''
    NR = rays.size()[0]

    A, B, C = einops.repeat(triangle, "pts dims -> pts NR dims", NR=NR)

    O, D = rays.unbind(1)

    mat = t.stack([- D, B - A, C - A], dim=-1)

    dets = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    vec = O - A

    sol = t.linalg.solve(mat, vec)
    s, u, v = sol.unbind(dim=-1)
    sumthn = ((s >=0 ) & (u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular)
    return ((s >=0 ) & (u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular)


intersects = raytrace_triangle_with_bug(rays2d, test_triangle)
img = intersects.reshape(num_pixels_y, num_pixels_z).int()
imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")
# %%
if MAIN:
    with open(section_dir / "pikachu.pt", "rb") as f:
        triangles = t.load(f)
# %%
def raytrace_mesh(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"]
) -> Float[Tensor, "nrays"]:
    '''
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    '''
    # Solving for (u,v,s) in A + u(B-A) + v(C-A) = O + sD
    # { [(B-A) (C-A) (-D)] [u v s] }_x = { [O - A] }_x
    # Where A,B,C,D,O are all vectors in three dimensions, we will stack the matricies along their last component
    # to get component-wise operations (B-A)_x * u + (C-A)_x * v + (-D)_x * s = (O-A)_x

    nrays = rays.shape[0]
    ntris = triangles.shape[0]
    big_triangle = einops.repeat(triangles, "ntris pts dims -> repeat ntris pts dims", repeat=nrays)
    big_rays = einops.repeat(rays, "nrays pts dims -> nrays repeat pts dims", repeat=ntris)
    A, B, C = big_triangle.unbind(dim=-2) # shape(nrays, ntris, dims) for each triangle coordinate
    O, D = big_rays.unbind(dim=-2) # shape(nrays, ntris, dims) for origin and vector
    x1 = (B-A)
    x2 = (C-A)
    x3 = (-D)
    batch_sol = (O-A) # shape(nrays, ntris, dims)
    batch_eqs = t.stack((x1,x2,x3), dim=-1) # collect [dims, variables] matrix, so that each row is a dimension. shape(nrays, ntris, dims, variables)
    zero_det_filter = t.linalg.det(batch_eqs).abs() < 1e-8
    batch_eqs[zero_det_filter] = t.eye(3)
    sol = t.linalg.solve(batch_eqs, batch_sol)
    u, v, s = sol.unbind(dim=-1) # each is size(nrays, ntris), create mask so that they are in the correct range
    hits_triangle_mask = t.all(sol >= 0, dim=-1) & (u+v <= 1) & ~(zero_det_filter) # shape(nrays, ntris)
    s_broadcast = einops.repeat(s, "nrays ntris -> nrays ntris 3")
    hits_rays = ((s_broadcast * D) + O)
    rays_distances = t.linalg.vector_norm(hits_rays, dim=-1)
    rays_distances[~hits_triangle_mask] = t.inf
    min_ray_dist = t.min(rays_distances, dim=-1).values #min distance over triangles
    return min_ray_dist


if MAIN:
    num_pixels_y = 120
    num_pixels_z = 120
    y_limit = z_limit = 1

    rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
    rays[:, 0] = t.tensor([-2, 0.0, 0.0])
    dists = raytrace_mesh(rays, triangles)
    intersects = t.isfinite(dists).view(num_pixels_y, num_pixels_z)
    dists_square = dists.view(num_pixels_y, num_pixels_z)
    img = t.stack([intersects, dists_square], dim=0)

    fig = px.imshow(img, facet_col=0, origin="lower", color_continuous_scale="magma", width=1000)
    fig.update_layout(coloraxis_showscale=False)
    for i, text in enumerate(["Intersects", "Distance"]): 
        fig.layout.annotations[i]['text'] = text
    fig.show()
# %%
def raytrace_mesh(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"]
) -> Float[Tensor, "nrays"]:
    '''
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    '''
    # SOLUTION
    NR = rays.size(0)
    NT = triangles.size(0)

    # Each triangle is [[Ax, Ay, Az], [Bx, By, Bz], [Cx, Cy, Cz]]
    triangles = einops.repeat(triangles, "NT pts dims -> pts NR NT dims", NR=NR)
    A, B, C = triangles
    assert A.shape == (NR, NT, 3)

    # Each ray is [[Ox, Oy, Oz], [Dx, Dy, Dz]]
    rays = einops.repeat(rays, "NR pts dims -> pts NR NT dims", NT=NT)
    O, D = rays
    assert O.shape == (NR, NT, 3)

    # Define matrix on left hand side of equation
    mat: Float[Tensor, "NR NT 3 3"] = t.stack([- D, B - A, C - A], dim=-1)
    # Get boolean of where matrix is singular, and replace it with the identity in these positions
    dets: Float[Tensor, "NR NT"] = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    # Define vector on the right hand side of equation
    vec: Float[Tensor, "NR NT 3"] = O - A

    # Solve eqns (note, s is the distance along ray)
    sol: Float[Tensor, "NR NT 3"] = t.linalg.solve(mat, vec)
    s, u, v = sol.unbind(-1)

    # Get boolean of intersects, and use it to set distance to infinity wherever there is no intersection
    intersects = ((u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular)
    s[~intersects] = float("inf") # t.inf

    # Get the minimum distance (over all triangles) for each ray
    return s.min(dim=-1).values

if MAIN:
    num_pixels_y = 120
    num_pixels_z = 120
    y_limit = z_limit = 1

    rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
    rays[:, 0] = t.tensor([-2, 0.0, 0.0])
    dists = raytrace_mesh(rays, triangles)
    intersects = t.isfinite(dists).view(num_pixels_y, num_pixels_z)
    dists_square = dists.view(num_pixels_y, num_pixels_z)
    img = t.stack([intersects, dists_square], dim=0)

    fig = px.imshow(img, facet_col=0, origin="lower", color_continuous_scale="magma", width=1000)
    fig.update_layout(coloraxis_showscale=False)
    for i, text in enumerate(["Intersects", "Distance"]): 
        fig.layout.annotations[i]['text'] = text
    fig.show()