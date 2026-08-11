"""Pure Python reimplementation of SOAX (Stretching Open Active Contours).

Replaces the C++ TSOAX binary which requires Qt5/VTK/X11 display server.
Works on headless servers, integrates directly into Optuna optimization.

Algorithm (Xu et al., Scientific Reports 2015):
    1. Initialize snakes from skeleton branches of binary mask
    2. Evolve each snake by minimizing E = E_int + E_ext:
       - E_int = alpha*|r_s|^2 + beta*|r_ss|^2 (stretching + bending)
       - E_ext = external_factor * image_gradient + stretch_factor * tip_force
       Solved as: (A + gamma*I) * x_new = gamma * x_old + F_ext
       where A is pentadiagonal (from alpha, beta boundary conditions)
    3. Link fragmented filaments using distance + angle criteria

Input: binary mask (uint8, 0/255)
Output: list of instance dicts with 'centerline' (row, col), 'length', 'area'

Usage:
    from pysoax import extract_soax_instances
    instances = extract_soax_instances(binary_mask, params)
"""

import numpy as np
from scipy import ndimage
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


# ─── Default SOAX parameters ───────────────────────────

DEFAULT_PARAMS = {
    "gaussian_std": 0.693,
    "ridge_threshold": 0.0178,
    "stretch_factor": 0.998,
    "alpha": 0.25,
    "beta": 0.06,
    "gamma": 4.597,
    "external_factor": 1.918,
    "min_snake_length": 13,
    "max_iterations": 5000,
    "change_threshold": 0.001,
    "check_period": 100,
    "point_spacing": 1.0,
    "grouping_distance": 4.018,
    "direction_threshold": 0.87,  # cos(~30 deg)
}


# ─── Pentadiagonal matrix for open snake ────────────────

def _build_pentadiag_open(n, alpha, beta, gamma):
    """Build pentadiagonal banded matrix for open active contour.

    For open boundary conditions (free endpoints), the first/last 2 rows
    have modified coefficients.

    Returns (5, n) array for scipy.linalg.solve_banded with (2, 2) bands.
    """
    a = alpha
    b = beta

    # Main diagonal
    d0 = np.full(n, 2 * a + 6 * b + gamma)
    # Sub/super diagonal ±1
    d1 = np.full(n, -a - 4 * b)
    # Sub/super diagonal ±2
    d2 = np.full(n, b)

    # Open boundary conditions (free endpoints):
    # First point
    d0[0] = a + b + gamma
    d1[0] = -2 * b           # d1[0] is the +1 entry of row 0
    # Second point
    d0[1] = a + 5 * b + gamma
    # Last point
    d0[-1] = a + b + gamma
    d1[-1] = -2 * b          # d1[-1] is the -1 entry of last row
    # Second-to-last
    d0[-2] = a + 5 * b + gamma

    # Pack into banded format (ab[i, j] = A[i-2+j, j] for upper=2)
    ab = np.zeros((5, n))
    ab[0, 2:] = d2[:-2]      # +2 diagonal
    ab[1, 1:] = d1[:-1]      # +1 diagonal
    ab[2, :] = d0             # main diagonal
    ab[3, :-1] = d1[:-1]     # -1 diagonal
    ab[4, :-2] = d2[:-2]     # -2 diagonal

    return ab


# ─── Snake evolution ────────────────────────────────────

def _compute_image_force(grad_r, grad_c, points):
    """Interpolate image gradient at snake point positions.

    Args:
        grad_r: (H, W) gradient in row direction.
        grad_c: (H, W) gradient in column direction.
        points: (N, 2) snake points (row, col).

    Returns:
        (N, 2) force vectors (row, col).
    """
    coords = points.T  # (2, N) for map_coordinates
    fr = ndimage.map_coordinates(grad_r, coords, order=1, mode="nearest")
    fc = ndimage.map_coordinates(grad_c, coords, order=1, mode="nearest")
    return np.column_stack([fr, fc])


def _stretching_tip_force(points, image, stretch_factor, fg_mean, bg_mean):
    """Compute stretching force at snake endpoints.

    At each tip, force = stretch_factor * (I(tip) - mean) / (fg - bg) * tangent.
    Positive force (tip on filament) → extend. Negative → retract.

    Args:
        points: (N, 2) snake points.
        image: (H, W) smoothed image.
        stretch_factor: force magnitude.
        fg_mean: mean foreground intensity.
        bg_mean: mean background intensity.

    Returns:
        head_force: (2,) force at first point.
        tail_force: (2,) force at last point.
    """
    H, W = image.shape
    denom = max(abs(fg_mean - bg_mean), 1e-6)

    forces = np.zeros((len(points), 2))

    # Head (first point): tangent points outward (away from snake body)
    if len(points) >= 2:
        tangent = points[0] - points[1]
        norm = np.linalg.norm(tangent)
        if norm > 1e-6:
            tangent /= norm
        r, c = int(np.clip(points[0, 0], 0, H-1)), int(np.clip(points[0, 1], 0, W-1))
        intensity = image[r, c]
        strength = stretch_factor * (intensity - bg_mean) / denom
        forces[0] = strength * tangent

    # Tail (last point)
    if len(points) >= 2:
        tangent = points[-1] - points[-2]
        norm = np.linalg.norm(tangent)
        if norm > 1e-6:
            tangent /= norm
        r, c = int(np.clip(points[-1, 0], 0, H-1)), int(np.clip(points[-1, 1], 0, W-1))
        intensity = image[r, c]
        strength = stretch_factor * (intensity - bg_mean) / denom
        forces[-1] = strength * tangent

    return forces


def _resample_snake(points, spacing=1.0):
    """Resample snake to uniform point spacing via cubic spline.

    Args:
        points: (N, 2) snake points.
        spacing: target distance between consecutive points.

    Returns:
        (M, 2) resampled points with ~uniform spacing.
    """
    if len(points) < 3:
        return points

    # Compute arc length
    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    arc = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_length = arc[-1]

    if total_length < spacing:
        return points

    # Cubic spline interpolation
    try:
        cs_r = CubicSpline(arc, points[:, 0], bc_type="natural")
        cs_c = CubicSpline(arc, points[:, 1], bc_type="natural")
    except ValueError:
        return points

    n_new = max(2, int(np.ceil(total_length / spacing)) + 1)
    arc_new = np.linspace(0, total_length, n_new)

    return np.column_stack([cs_r(arc_new), cs_c(arc_new)])


def evolve_snake(points, image_smooth, grad_r, grad_c, params,
                 fg_mean, bg_mean):
    """Evolve a single open active contour.

    Args:
        points: (N, 2) initial snake points (row, col).
        image_smooth: (H, W) Gaussian-smoothed image.
        grad_r, grad_c: (H, W) image gradient fields.
        params: SOAX parameter dict.
        fg_mean: mean foreground intensity.
        bg_mean: mean background intensity.

    Returns:
        (M, 2) converged snake points.
    """
    from scipy.linalg import solve_banded

    alpha = params.get("alpha", 0.25)
    beta = params.get("beta", 0.06)
    gamma = params.get("gamma", 4.597)
    ext_factor = params.get("external_factor", 1.918)
    stretch = params.get("stretch_factor", 0.998)
    # For binary masks, 500 iterations is sufficient (no ridge hunting needed)
    max_iter = params.get("max_iterations", 500)
    check_period = params.get("check_period", 20)
    change_thr = params.get("change_threshold", 0.01)
    spacing = params.get("point_spacing", 1.0)

    pts = points.copy().astype(np.float64)
    H, W = image_smooth.shape

    # Pre-build matrix if snake length is stable (avoid rebuilding every iter)
    n = len(pts)
    ab = _build_pentadiag_open(n, alpha, beta, gamma)

    for iteration in range(max_iter):
        if len(pts) != n:
            # Snake length changed after resampling
            n = len(pts)
            if n < 3:
                break
            ab = _build_pentadiag_open(n, alpha, beta, gamma)

        # Image gradient force
        img_force = _compute_image_force(grad_r, grad_c, pts)

        # Stretching tip force
        tip_force = _stretching_tip_force(
            pts, image_smooth, stretch, fg_mean, bg_mean)

        # Right-hand side: gamma * x_old + external forces
        rhs_r = gamma * pts[:, 0] + ext_factor * img_force[:, 0] + tip_force[:, 0]
        rhs_c = gamma * pts[:, 1] + ext_factor * img_force[:, 1] + tip_force[:, 1]

        # Solve banded system
        new_r = solve_banded((2, 2), ab, rhs_r)
        new_c = solve_banded((2, 2), ab, rhs_c)

        new_pts = np.column_stack([new_r, new_c])

        # Clamp to image bounds
        new_pts[:, 0] = np.clip(new_pts[:, 0], 0, H - 1)
        new_pts[:, 1] = np.clip(new_pts[:, 1], 0, W - 1)

        # Check convergence periodically
        if (iteration + 1) % check_period == 0:
            max_disp = np.max(np.linalg.norm(new_pts - pts, axis=1))
            if max_disp < change_thr:
                pts = new_pts
                break
            # Resample to maintain uniform spacing
            pts = _resample_snake(new_pts, spacing)
        else:
            pts = new_pts

    return pts


# ─── Skeleton-based initialization ──────────────────────

def _extract_branches(skeleton, min_branch_length=5):
    """Extract individual branches from skeleton image.

    A branch is a connected path between junction/endpoint pixels.
    Uses direct 8-connected pixel tracing (fast, no KDTree).

    Args:
        skeleton: (H, W) binary skeleton.
        min_branch_length: skip branches shorter than this (pixels).

    Returns:
        list of (N, 2) arrays, each a branch as (row, col) points.
    """
    skel = skeleton.astype(bool)
    H, W = skel.shape

    # Find junction pixels using neighbor count
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    neighbors = ndimage.convolve(skel.astype(np.uint8), kernel, mode="constant")
    neighbors *= skel

    junctions = skel & (neighbors >= 3)

    # Remove junctions → label connected components
    skel_no_junc = skel & ~junctions
    labeled, n_components = ndimage.label(skel_no_junc)

    # Fast branch extraction: for each component, get sorted pixel coords
    branches = []

    # Use ndimage.find_objects for fast per-component extraction
    slices = ndimage.find_objects(labeled)
    for comp_id, slc in enumerate(slices, 1):
        if slc is None:
            continue
        comp_mask = labeled[slc] == comp_id
        n_pixels = comp_mask.sum()
        if n_pixels < min_branch_length:
            continue

        # Get coordinates relative to slice, then offset
        local_coords = np.argwhere(comp_mask)
        coords = local_coords + np.array([slc[0].start, slc[1].start])

        # Order by tracing 8-connected path through the mask
        ordered = _trace_connected_path(coords, comp_mask, slc)
        if ordered is not None and len(ordered) >= min_branch_length:
            branches.append(ordered.astype(np.float64))

    return branches


def _trace_connected_path(coords, local_mask, slc):
    """Trace 8-connected path through a thin (1px wide) branch.

    Uses the local mask directly for O(1) neighbor lookup.

    Args:
        coords: (N, 2) global coordinates.
        local_mask: (h, w) binary mask of the branch (in slice coords).
        slc: tuple of slices for offset computation.

    Returns:
        (N, 2) ordered global coordinates, or None.
    """
    h, w = local_mask.shape
    if local_mask.sum() < 2:
        return None

    # Work in local coordinates
    visited = np.zeros_like(local_mask, dtype=bool)

    # Find an endpoint (pixel with only 1 neighbor in component)
    local_coords = np.argwhere(local_mask)

    # Pick start: find pixel with fewest neighbors (endpoint)
    best_start = local_coords[0]
    min_nb = 8
    for lc in local_coords[:20]:  # check first 20, usually enough
        r, c = lc
        nb = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and local_mask[nr, nc]:
                    nb += 1
        if nb < min_nb:
            min_nb = nb
            best_start = lc
            if nb == 1:
                break

    # Trace path
    path = []
    r, c = best_start
    visited[r, c] = True
    path.append((r + slc[0].start, c + slc[1].start))

    for _ in range(local_mask.sum()):
        found = False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and local_mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    path.append((nr + slc[0].start, nc + slc[1].start))
                    r, c = nr, nc
                    found = True
                    break
            if found:
                break
        if not found:
            break

    if len(path) < 2:
        return None
    return np.array(path)


# ─── Main extraction pipeline ───────────────────────────

def extract_soax_instances(binary_mask, params=None,
                            cos2theta=None, sin2theta=None,
                            embeddings=None):
    """Extract filament instances from binary mask.

    Two modes:
        - Graph mode (default): Skeleton → graph → longest paths through junctions.
          Fast (~0.5s), good for binary masks. No snake evolution needed.
        - Evolution mode (evolve=True): Additionally refines paths with active
          contour evolution on distance transform.

    Args:
        binary_mask: (H, W) uint8 array (0 or 255) or bool.
        params: SOAX parameter dict (uses DEFAULT_PARAMS if None).
            Special keys:
            - `orient_weight` (float, default 0.0): if > 0 and
              cos2theta/sin2theta are provided, the junction tracer uses
              the model's orientation field to disambiguate ambiguous
              junctions.
            - `embed_weight` (float, default 0.0): if > 0 and embeddings
              are provided, the junction tracer uses the model's
              instance-identity embeddings as an additional disambiguator.
              Unlike orientation (averaged in GT), embeddings preserve per-
              MT identity at crossings.
        cos2theta, sin2theta: optional (H, W) model orientation arrays
            (tanh-activated). When provided with params['orient_weight'] > 0,
            enables orientation-aware junction tracing.
        embeddings: optional (E, H, W) model embedding array (raw, not L2-
            normalized). When provided with params['embed_weight'] > 0,
            enables embedding-aware junction tracing.

    Returns:
        list of instance dicts with keys:
            'centerline': (N, 2) array in (row, col) format
            'area': int (number of centerline points)
            'length': float (arc length in pixels)
            'mask': None (centerlines only, like original SOAX)
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    H, W = binary_mask.shape
    min_length = params.get("min_snake_length", 13)
    orient_weight = float(params.get("orient_weight", 0.0))
    embed_weight = float(params.get("embed_weight", 0.0))

    # Normalize to bool
    fg_mask = binary_mask > (127 if binary_mask.max() > 1 else 0.5)

    # Skeletonize
    skeleton = skeletonize(fg_mask)
    if skeleton.sum() < min_length:
        return []

    # Extract filament paths using graph-based traversal.
    # When orient_weight > 0 and orientation field is supplied, the tracer
    # uses doubled-angle alignment with the model output as a tie-breaker
    # at junction pixels. When embed_weight > 0 and embeddings are supplied,
    # additional instance-identity cosine similarity is combined into the
    # junction score.
    paths = _extract_paths_from_skeleton(
        skeleton, min_length,
        cos2theta=cos2theta, sin2theta=sin2theta,
        orient_weight=orient_weight,
        embeddings=embeddings, embed_weight=embed_weight)

    if not paths:
        return []

    # Optional: refine paths with snake evolution on distance transform.
    # Default-OFF: while C++ SOAX uses evolution, our distance transform on
    # 1-px-wide skeletons has a degenerate gradient field that can collapse
    # parallel filaments into a single distorted snake. On real CVAT data
    # (where masks are 3-5 px wide) the gain is marginal (+0.001 F1@10),
    # not worth the pathology risk on edge cases. Set evolve=True
    # explicitly in params to enable.
    if params.get("evolve", False):
        mask_float = fg_mask.astype(np.float64)
        sigma = params.get("gaussian_std", 0.693)
        image_smooth = ndimage.gaussian_filter(mask_float, sigma=sigma)
        dist_transform = ndimage.distance_transform_edt(fg_mask)
        grad_r = ndimage.sobel(dist_transform, axis=0)
        grad_c = ndimage.sobel(dist_transform, axis=1)
        fg_mean = image_smooth[fg_mask].mean() if fg_mask.any() else 0.5
        bg_mean = image_smooth[~fg_mask].mean() if (~fg_mask).any() else 0.0

        evo_params = params.copy()
        evo_params["stretch_factor"] = 0.0  # no tip stretching for binary masks

        refined_paths = []
        for path in paths:
            if len(path) >= 3:
                path = _resample_snake(path, params.get("point_spacing", 1.0))
                if len(path) >= 3:
                    path = evolve_snake(
                        path, image_smooth, grad_r, grad_c, evo_params,
                        fg_mean, bg_mean)
            refined_paths.append(path)
        paths = refined_paths

    # Optional fragment linking pass (Stage A2). The historically dead
    # _link_and_output is now Hungarian-based and globally optimal, but on
    # v6 e25 it does not improve F1@10 because the dominant failure mode
    # is wrong-branch path tracing at junctions, not fragmentation.
    # Linking is opt-in via params['enable_linking']=True for cases where
    # the model produces noticeably fragmented predictions.
    if params.get("enable_linking", False):
        grouping_dist = params.get("grouping_distance", 4.018)
        direction_thr = params.get("direction_threshold", 0.87)
        if len(paths) >= 2 and grouping_dist > 0:
            return _link_and_output(paths, grouping_dist, direction_thr,
                                    min_length)

    return _snakes_to_instances(paths, min_length)


def _extract_paths_from_skeleton(skeleton, min_length=13,
                                  cos2theta=None, sin2theta=None,
                                  orient_weight=0.0,
                                  embeddings=None, embed_weight=0.0):
    """Extract filament paths from skeleton using graph traversal.

    Instead of fragmenting at junctions, traces longest paths through
    the skeleton graph from endpoint to endpoint (or endpoint to junction).

    At junctions, greedily follows the straightest continuation
    (smallest angle change). When the model's orientation field is provided
    via cos2theta/sin2theta, junction selection is additionally weighted
    by the doubled-angle alignment between candidate pixel direction and
    the model's tangent estimate at the candidate pixel. When the model's
    embedding field is provided via embeddings + embed_weight, junction
    selection also considers instance-identity cosine similarity.

    Args:
        skeleton: (H, W) binary skeleton image.
        min_length: minimum path length in pixels.
        cos2theta, sin2theta: optional (H, W) model orientation field.
        orient_weight: weight for orient-field junction term (0=disabled).
        embeddings: optional (E, H, W) model embedding field.
        embed_weight: weight for embedding-identity junction term (0=disabled).

    Returns:
        list of (N, 2) float64 arrays (row, col paths).
    """
    skel = skeleton.astype(bool)
    H, W = skel.shape

    # Classify pixels by neighbor count
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    neighbors = ndimage.convolve(skel.astype(np.uint8), kernel, mode="constant")
    neighbors *= skel

    endpoint_mask = skel & (neighbors == 1)
    junction_mask = skel & (neighbors >= 3)

    endpoints = list(map(tuple, np.argwhere(endpoint_mask)))

    # Track which pixels have been used
    used = np.zeros_like(skel, dtype=bool)

    paths = []

    # Trace from each endpoint (greedy: start from endpoints, not junctions)
    for start in endpoints:
        if used[start]:
            continue
        path = _trace_path(start, skel, used, junction_mask, H, W,
                           cos2theta=cos2theta, sin2theta=sin2theta,
                           orient_weight=orient_weight,
                           embeddings=embeddings, embed_weight=embed_weight)
        if path is not None and len(path) >= min_length:
            paths.append(np.array(path, dtype=np.float64))

    # Trace remaining unused skeleton pixels (loops, isolated segments)
    remaining = skel & ~used
    labeled, n_comp = ndimage.label(remaining)
    for comp_id in range(1, n_comp + 1):
        comp_coords = np.argwhere(labeled == comp_id)
        if len(comp_coords) < min_length:
            continue
        # Trace from first pixel
        start = tuple(comp_coords[0])
        path = _trace_path(start, remaining, used, junction_mask, H, W,
                           cos2theta=cos2theta, sin2theta=sin2theta,
                           orient_weight=orient_weight)
        if path is not None and len(path) >= min_length:
            paths.append(np.array(path, dtype=np.float64))

    return paths


def _trace_path(start, skeleton, used, junction_mask, H, W,
                 cos2theta=None, sin2theta=None, orient_weight=0.0,
                 embeddings=None, embed_weight=0.0, embed_window=15):
    """Trace a single path from start pixel through skeleton.

    At junctions, follows the straightest continuation (smallest curvature).
    Stops at: dead end, already-used pixel, or image boundary.

    Args:
        start: (row, col) starting pixel.
        skeleton: (H, W) bool skeleton.
        used: (H, W) bool mask of already-traced pixels (modified in-place).
        junction_mask: (H, W) bool junction pixels.
        H, W: image dimensions.
        cos2theta, sin2theta: optional (H, W) float arrays from the model's
            orientation head (tanh-activated). When provided together with
            orient_weight > 0, the junction selection score becomes:
                score = cos_angle + orient_weight * |orient_align|
            where orient_align is the doubled-angle dot product between the
            candidate's pixel direction and the model's orientation field at
            the candidate pixel. This lets the tracer disambiguate junctions
            using the global tangent estimate the model already learned.
        orient_weight: scalar weight for the orientation alignment term
            (typical values: 0.5–1.5). 0.0 disables orient-aware tracing.
        embeddings: optional (E, H, W) float array from the model's embedding
            head. When provided with embed_weight > 0, the tracer uses
            instance-identity cosine similarity to disambiguate at junctions.
            Unlike the orientation field, embeddings are NOT averaged across
            overlapping instances in GT generation, so they preserve per-MT
            identity at crossings — exactly where orient fails structurally.
        embed_weight: scalar weight for the embedding alignment term. 0.0
            disables. Moderate values 0.5-1.5 recommended given embeddings
            are only partially discriminative (v6 e25 diagnostic showed
            ratio ≈ 19 mean but some close-pair instances below delta_d).
        embed_window: number of most-recent path pixels used to compute the
            running mean embedding. Longer window = more stable signature
            but slower adaptation to MT curvature. Default 15 ≈ 1 MT radius
            of samples at 1px spacing.

    Returns:
        list of (row, col) tuples, or None.
    """
    use_orient = (cos2theta is not None and sin2theta is not None
                  and orient_weight > 0.0)
    use_embed = (embeddings is not None and embed_weight > 0.0)

    path = [start]
    used[start] = True
    current = start
    prev = None

    # Running embedding "signature" of the trace — mean embedding over the
    # last `embed_window` path pixels. Rebuilt on demand when needed.
    # Storing the mean directly (not the full list) keeps the fast path fast;
    # recomputing from `path[-embed_window:]` at each junction is cheap (≤15
    # lookups + mean) and avoids stale data when the path curves sharply.
    E_dim = embeddings.shape[0] if use_embed else 0

    max_path_len = max(H, W) * 3  # no path longer than 3× image dimension
    for _ in range(max_path_len):
        r, c = current

        # Find 8-connected skeleton neighbors that haven't been used
        candidates = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and skeleton[nr, nc]:
                    if not used[nr, nc]:
                        candidates.append((nr, nc))

        if not candidates:
            break

        if len(candidates) == 1:
            next_pixel = candidates[0]
        else:
            # Choose straightest continuation (smallest angle change)
            if prev is not None:
                direction = np.array(current) - np.array(prev)
                dir_norm = np.linalg.norm(direction)
                if dir_norm > 0:
                    direction = direction / dir_norm
                else:
                    direction = np.array([1.0, 0.0])
            else:
                direction = np.array([1.0, 0.0])

            # Running path-mean embedding — computed only at junctions (cheap).
            # Skips if no path history (we're at the very start).
            path_embed_mean = None
            path_embed_norm = 0.0
            if use_embed and len(path) >= 2:
                window_start = max(0, len(path) - embed_window)
                window_pts = path[window_start:]
                window_r = np.array([p[0] for p in window_pts])
                window_c = np.array([p[1] for p in window_pts])
                # Gather E-dim vectors then average
                win_embeds = embeddings[:, window_r, window_c]  # (E, K)
                path_embed_mean = win_embeds.mean(axis=1)  # (E,)
                path_embed_norm = float(np.linalg.norm(path_embed_mean))

            best_score = -1e9
            next_pixel = candidates[0]
            for cand in candidates:
                cand_dir = np.array(cand) - np.array(current)
                cand_norm = np.linalg.norm(cand_dir)
                if cand_norm <= 0:
                    continue
                cand_unit = cand_dir / cand_norm
                cos_angle = float(np.dot(direction, cand_unit))

                score = cos_angle
                if use_orient:
                    # Convert candidate's pixel direction (dr, dc) into
                    # doubled-angle representation: with path stored as
                    # (x,y) = (col,row), tangent = (dx,dy) = (dc,dr), so
                    # cos(θ)=dc, sin(θ)=dr, hence cos(2θ)=2dc²-1, sin(2θ)=2·dr·dc.
                    cdr, cdc = float(cand_unit[0]), float(cand_unit[1])
                    cos2_phi = 2.0 * cdc * cdc - 1.0
                    sin2_phi = 2.0 * cdr * cdc

                    # Sample orient field FURTHER DOWNSTREAM (6px) along the
                    # candidate direction. The GT generation averages
                    # orientation at overlapping pixels (generate_4ch_gt.py
                    # `cos2theta_sum / orient_count`), so the model's
                    # prediction in junction zones is structurally an
                    # average and useless as a disambiguator. We need to
                    # walk OUT of the junction zone to where only one MT
                    # contributes — empirically ~5–7 px is enough.
                    look_ahead = 6
                    sample_r = int(round(cand[0] + cdr * look_ahead))
                    sample_c = int(round(cand[1] + cdc * look_ahead))
                    sample_r = max(0, min(H - 1, sample_r))
                    sample_c = max(0, min(W - 1, sample_c))
                    c2t = float(cos2theta[sample_r, sample_c])
                    s2t = float(sin2theta[sample_r, sample_c])
                    orient_align = cos2_phi * c2t + sin2_phi * s2t
                    # |.| because doubled angle is direction-agnostic.
                    score += orient_weight * abs(orient_align)

                if use_embed and path_embed_mean is not None and path_embed_norm > 1e-6:
                    # Same look-ahead trick as orient: sample downstream of
                    # the candidate to escape the junction zone where two
                    # instance identities overlap. Embeddings ARE NOT
                    # averaged in GT (each pixel has one instance id), but
                    # the model's learned embedding may still be "smeared"
                    # in junction neighborhoods from receptive field
                    # overlap. 6 px out the dominant continuation's
                    # embedding wins.
                    cdr, cdc = float(cand_unit[0]), float(cand_unit[1])
                    look_ahead = 6
                    sample_r = int(round(cand[0] + cdr * look_ahead))
                    sample_c = int(round(cand[1] + cdc * look_ahead))
                    sample_r = max(0, min(H - 1, sample_r))
                    sample_c = max(0, min(W - 1, sample_c))
                    cand_embed = embeddings[:, sample_r, sample_c]  # (E,)
                    cand_norm_e = float(np.linalg.norm(cand_embed))
                    if cand_norm_e > 1e-6:
                        # Cosine similarity ∈ [-1, 1]. High = same instance.
                        cos_embed = float(np.dot(path_embed_mean, cand_embed)
                                          / (path_embed_norm * cand_norm_e))
                        score += embed_weight * cos_embed

                if score > best_score:
                    best_score = score
                    next_pixel = cand

        prev = current
        current = next_pixel
        path.append(current)
        used[current] = True

    return path if len(path) >= 2 else None


def _link_and_output(snakes, grouping_dist, direction_thr, min_length,
                      max_length=5000, angle_weight=2.0):
    """Link snake fragments via globally-optimal Hungarian matching.

    Replaces the original greedy sort-by-distance loop with a proper
    minimum-cost matching on the bipartite graph of endpoints. This fixes
    the order-dependence pathology where greedy locks in suboptimal
    short-distance pairs that conflict with longer-distance but globally
    better matches.

    Algorithm:
        1. Collect head + tail endpoints from each snake (2N total).
        2. Build N_ep × N_ep cost matrix:
           cost[i,j] = ||p_i - p_j|| + α·(1 - |cos(d_i, d_j)|)
           with cost set to inf if forbidden (same snake, distance >
           grouping_dist, or |cos(d_i, d_j)| < direction_thr).
        3. Solve via scipy.optimize.linear_sum_assignment (Jonker-Volgenant
           variant; handles inf entries gracefully).
        4. The cost matrix is symmetric, so each true match appears as
           both (i,j) and (j,i) in the result. Iterate with i < j and
           perform the merge in cost-ascending order to allow chain
           consolidation across multiple merges.

    Args:
        snakes: list of (N, 2) arrays (evolved snake points).
        grouping_dist: max distance for linking endpoints.
        direction_thr: min |cos(angle)| for angular consistency.
        min_length: minimum filament length after linking.
        max_length: maximum filament length (prevents runaway chaining).
        angle_weight: penalty multiplier for angular mismatch in cost
            (typical 1.0–3.0; balances px-distance against unitless cos).

    Returns:
        list of instance dicts.
    """
    if not snakes:
        return []

    # Collect all endpoints with direction vectors
    endpoints = []  # (snake_idx, end_type, position, direction)
    for i, snake in enumerate(snakes):
        if len(snake) < 2:
            continue
        head_dir = snake[0] - snake[min(3, len(snake) - 1)]
        norm = np.linalg.norm(head_dir)
        if norm > 1e-6:
            head_dir /= norm
        endpoints.append((i, "head", snake[0], head_dir))

        tail_dir = snake[-1] - snake[max(-4, -len(snake))]
        norm = np.linalg.norm(tail_dir)
        if norm > 1e-6:
            tail_dir /= norm
        endpoints.append((i, "tail", snake[-1], tail_dir))

    if len(endpoints) < 2:
        return _snakes_to_instances(snakes, min_length)

    # Build cost matrix. Use a large finite "forbidden" value rather than
    # +inf so linear_sum_assignment never picks these by accident; we still
    # filter post-hoc by comparing to the threshold.
    n_ep = len(endpoints)
    forbidden = 1e9
    cost = np.full((n_ep, n_ep), forbidden, dtype=np.float64)

    positions = np.array([ep[2] for ep in endpoints])
    directions = np.array([ep[3] for ep in endpoints])

    # Pre-filter via KDTree to keep cost-matrix construction O(K) rather
    # than O(N²) for large N. Each endpoint only gets compared to spatial
    # neighbors within grouping_dist.
    tree = cKDTree(positions)
    pair_set = tree.query_pairs(r=grouping_dist)
    for i, j in pair_set:
        si = endpoints[i][0]
        sj = endpoints[j][0]
        if si == sj:
            continue
        di = directions[i]
        dj = directions[j]
        cos_angle = abs(float(np.dot(di, dj)))
        if cos_angle < direction_thr:
            continue
        dist = float(np.linalg.norm(positions[i] - positions[j]))
        # Convex combination: distance dominates, angle penalizes only
        # mismatches (perfect alignment gives no penalty).
        c = dist + angle_weight * (1.0 - cos_angle)
        cost[i, j] = c
        cost[j, i] = c

    # Hungarian assignment
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost)

    # Collect candidate matches with cost < forbidden, dedup by i<j,
    # sort by cost ascending so chained merges go cheapest-first.
    matches = []
    for r, c in zip(row_ind, col_ind):
        if r >= c:
            continue
        if cost[r, c] >= forbidden:
            continue
        matches.append((cost[r, c], r, c))
    matches.sort()

    merge_map = {i: snakes[i] for i in range(len(snakes))}
    merged_into = {}

    for _, ei, ej in matches:
        si = endpoints[ei][0]
        sj = endpoints[ej][0]
        # Resolve merge chains
        while si in merged_into:
            si = merged_into[si]
        while sj in merged_into:
            sj = merged_into[sj]
        if si == sj:
            continue
        if si not in merge_map or sj not in merge_map:
            continue

        s1 = merge_map[si]
        s2 = merge_map[sj]

        d_hh = np.linalg.norm(s1[0] - s2[0])
        d_ht = np.linalg.norm(s1[0] - s2[-1])
        d_th = np.linalg.norm(s1[-1] - s2[0])
        d_tt = np.linalg.norm(s1[-1] - s2[-1])
        min_d = min(d_hh, d_ht, d_th, d_tt)
        if min_d > grouping_dist:
            continue

        if min_d == d_th:
            merged = np.vstack([s1, s2])
        elif min_d == d_ht:
            merged = np.vstack([s2, s1])
        elif min_d == d_tt:
            merged = np.vstack([s1, s2[::-1]])
        else:
            merged = np.vstack([s1[::-1], s2])

        diffs = np.diff(merged, axis=0)
        merged_len = float(np.sum(np.linalg.norm(diffs, axis=1)))
        if merged_len > max_length:
            continue

        merge_map[si] = merged
        del merge_map[sj]
        merged_into[sj] = si

    return _snakes_to_instances(list(merge_map.values()), min_length)


def _snakes_to_instances(snakes, min_length):
    """Convert snake point arrays to instance dicts.

    Args:
        snakes: list of (N, 2) arrays.
        min_length: minimum arc length.

    Returns:
        list of instance dicts.
    """
    instances = []
    for snake in snakes:
        if len(snake) < 2:
            continue
        diffs = np.diff(snake, axis=0)
        length = float(np.sum(np.linalg.norm(diffs, axis=1)))
        if length < min_length:
            continue
        instances.append({
            "centerline": snake.astype(np.float64),
            "area": len(snake),
            "length": length,
            "mask": None,
        })
    return instances


# ─── Convenience: run on image file ─────────────────────

def run_pysoax_on_image(image_path, params=None):
    """Run Python SOAX on a single binary mask image.

    Args:
        image_path: path to .tif binary mask (uint8, 0/255).
        params: SOAX parameter dict.

    Returns:
        list of instance dicts.
    """
    import tifffile
    mask = tifffile.imread(image_path)
    if mask.ndim > 2:
        mask = mask[0]
    return extract_soax_instances(mask, params)
