import numpy as np
import random
import os
import sys
sys.path.append('../utils')
import pickle
import ShapePrimitive
import ShapeOps
from shapely.geometry import Point
from shapely.affinity import scale, rotate, translate
from shapely.prepared import prep
import plot_utils

class XShapeGen:
    def __init__(self, config=None):
        self.config = config
        C = self.config['COMPLEX_SHAPE_GEN']
        # Seeds
        np.random.seed(C['random_seed_id'])
        random.seed(C['random_seed_id'])

        # Primitives live in canvas space
        self.ps = ShapePrimitive.PrimitiveShapes(
            canvas_size=C['canvas_size']
        )

        self.get_shape_operations()
        self.get_shape_primitives()

        self.binary_op = ShapeOps.BinaryShapeOp()
        self.unary_op = ShapeOps.UnaryShapeOp()
        self.unary_transform = ShapeOps.UnaryShapeTransform()

        # ----- Read ranges w/ fallbacks -----
        # Canvas-frame (your originals)
        self.canvas_translate_range = tuple(C.get('translate_range', [-10, 10]))
        self.canvas_rotate_range    = tuple(C.get('rotate_range',    [-30,  30]))
        self.canvas_scale_range     = tuple(C.get('scale_range',     [0.5,  2.0]))

        # Unit-frame jitters (used in _rand_primitive_unit)
        self.unit_jitter_translate_range = tuple(C.get('unit_jitter_translate_range', [-0.1, 0.1]))
        self.unit_jitter_rotate_range    = tuple(C.get('unit_jitter_rotate_range',    [-45, 45]))
        self.unit_jitter_scale_range     = tuple(C.get('unit_jitter_scale_range',     [0.8, 1.2]))

        # Unit-frame unary “steps” during construction
        self.unit_step_translate_range = tuple(C.get('unit_step_translate_range', [-0.08, 0.08]))
        self.unit_step_rotate_range    = tuple(C.get('unit_step_rotate_range',    [-30, 30]))
        self.unit_step_scale_range     = tuple(C.get('unit_step_scale_range',     [0.9, 1.1]))

        # Composition constraints (advanced)
        chg_lo, chg_hi = C.get('change_ratio_range', [0.12, 0.55])
        self.min_chg, self.max_chg = float(chg_lo), float(chg_hi)

        ov_lo, ov_hi = C.get('overlap_ratio_range', [0.25, 0.70])
        self.min_ov, self.max_ov = float(ov_lo), float(ov_hi)

        pa_lo, pa_hi = C.get('partner_area_frac_range', [0.25, 0.55])
        self.partner_area_frac_range = (float(pa_lo), float(pa_hi))

        self.destructive_min_area_frac = float(C.get('destructive_min_area_frac', 0.6))

    # ---------- config helpers ----------
    @staticmethod
    def _clean(g):
        try:
            return g.buffer(0)
        except Exception:
            return g

    @staticmethod
    def _is_valid_nonempty(g):
        return (g is not None) and (not g.is_empty) and g.is_valid

    @staticmethod
    def _is_connected_poly(g):
        g = XShapeGen._clean(g)
        if not XShapeGen._is_valid_nonempty(g):
            return False
        return g.geom_type == 'Polygon'

    @staticmethod
    def _area(g):
        try:
            return float(g.area)
        except Exception:
            return 0.0

    @staticmethod
    def _max_radius_about_origin(g):
        minx, miny, maxx, maxy = g.bounds
        cand = [(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)]
        try:
            if hasattr(g, "exterior") and g.exterior is not None:
                cand.extend(list(g.exterior.coords))
        except Exception:
            pass
        return max((np.hypot(x, y) for (x, y) in cand), default=1.0)

    # ---------- Fit to unit (no clipping) ----------
    def _to_unit_fit(self, g, margin=1e-3):
        g = self._clean(g)
        c = g.centroid
        g = translate(g, xoff=-c.x, yoff=-c.y)
        r = self._max_radius_about_origin(g)
        if r <= 0:
            r = 1.0
        s = (1.0 - margin) / r
        g = scale(g, xfact=s, yfact=s, origin=(0.0, 0.0))
        return self._clean(g)

    def _recenter_and_fit_if_needed(self, g, margin=1e-3):
        minx, miny, maxx, maxy = g.bounds
        if max(abs(minx), abs(maxx), abs(miny), abs(maxy)) <= 1.0 - margin:
            return g
        return self._to_unit_fit(g, margin=margin)

    # ---------- Ops wrappers (canvas-frame) ----------
    def apply_unary_op(self, shape, op_name):
        if op_name == 'scale':
            return self.unary_transform.run_transform(
                shape, 'scale',
                scale_x=np.random.uniform(*self.canvas_scale_range),
                scale_y=np.random.uniform(*self.canvas_scale_range)
            )
        elif op_name == 'translate':
            return self.unary_transform.run_transform(
                shape, 'translate',
                x=np.random.uniform(*self.canvas_translate_range),
                y=np.random.uniform(*self.canvas_translate_range)
            )
        elif op_name == 'rotate':
            return self.unary_transform.run_transform(
                shape, 'rotate',
                angle=np.random.uniform(*self.canvas_rotate_range)
            )

    def apply_binary_op(self, shape1, shape2, op_name):
        if op_name == 'union':
            return self.binary_op.run_op(shape1, shape2, 'union')
        elif op_name == 'intersect':
            return self.binary_op.run_op(shape1, shape2, 'intersect')
        elif op_name == 'subtract':
            return self.binary_op.run_op(shape1, shape2, 'subtract')
        elif op_name == 'xor':
            return self.binary_op.run_op(shape1, shape2, 'xor')
        elif op_name == 'convex_hull':
            return self.binary_op.run_op(shape1, shape2, 'convex_hull')

    # ---------- Primitives & ops ----------
    def get_shape_primitives(self):
        self.shape_primitives = [
            self.ps.circle(),
            self.ps.square(),
            self.ps.rectangle(),
            self.ps.triangle(),
            self.ps.ellipse(),
            self.ps.diamond(),
            self.ps.sector(),
            self.ps.pentagon(),
        ]

    def get_shape_operations(self):
        shape_ops = list()
        shape_ops.append({'op_name': 'scale', 'arity': 'unary'})
        shape_ops.append({'op_name': 'translate', 'arity': 'unary'})
        shape_ops.append({'op_name': 'rotate', 'arity': 'unary'})
        shape_ops.append({'op_name': 'union', 'arity': 'binary'})
        shape_ops.append({'op_name': 'intersect', 'arity': 'binary'})
        shape_ops.append({'op_name': 'subtract', 'arity': 'binary'})
        shape_ops.append({'op_name': 'xor', 'arity': 'binary'})
        shape_ops.append({'op_name': 'convex_hull', 'arity': 'binary'})
        self.shape_ops = shape_ops

    # ---------- New constructive generator (unit-fit, no clipping) ----------
    def create_shapes(self, binary_op_num, min_steps, max_steps):
        """
        Build a connected complex shape with exactly `binary_op_num` binary ops.
        Composition happens in a unit frame (fit-to-disk), never clipped to the disk.
        """
        partner_area_frac_range = self.partner_area_frac_range
        min_chg, max_chg = self.min_chg, self.max_chg
        min_ov,  max_ov  = self.min_ov,  self.max_ov
        destructive_min_area_frac = self.destructive_min_area_frac
        binary_ops_menu = ['union', 'intersect', 'subtract', 'xor', 'convex_hull']

        def _connectedify(candidate, anchor):
            cand = self._clean(candidate)
            if not self._is_valid_nonempty(cand):
                return cand
            if cand.geom_type == 'Polygon':
                return cand
            if cand.geom_type == 'MultiPolygon':
                parts = list(cand.geoms)
                touching = [p for p in parts if self._clean(p).intersects(self._clean(anchor))]
                pool = touching if touching else parts
                return max(pool, key=lambda g: g.area)
            return cand

        def _symmetric_diff_ratio(a, b):
            denom = self._area(a)
            if denom <= 1e-12:
                return 1.0
            return self._area(self._clean(a.symmetric_difference(b))) / denom

        def _overlap_ratio(a, b):
            denom = self._area(a)
            if denom <= 1e-12:
                return 0.0
            return self._area(self._clean(a.intersection(b))) / denom

        def _almost_equal(a, b, tol=0.95):
            inter = self._area(self._clean(a.intersection(b)))
            uni = self._area(self._clean(a.union(b)))
            if uni <= 1e-12:
                return True
            return (inter / uni) >= tol

        def _rand_primitive_unit():
            shp = np.random.choice(self.shape_primitives)
            shp = self._to_unit_fit(shp)
            # gentle unit-frame jitter using config
            for _ in range(np.random.randint(0, 3)):
                t = np.random.choice(['scale', 'rotate', 'translate'])
                if t == 'scale':
                    sx = np.random.uniform(*self.unit_jitter_scale_range)
                    sy = np.random.uniform(*self.unit_jitter_scale_range)
                    shp = scale(shp, xfact=sx, yfact=sy, origin='center')
                elif t == 'rotate':
                    ang = np.random.uniform(*self.unit_jitter_rotate_range)
                    shp = rotate(shp, angle=ang, origin='center')
                else:
                    dx = np.random.uniform(*self.unit_jitter_translate_range)
                    dy = np.random.uniform(*self.unit_jitter_translate_range)
                    shp = translate(shp, xoff=dx, yoff=dy)
                shp = self._recenter_and_fit_if_needed(self._clean(shp))
            return self._clean(shp)

        def _warp_partner_for_current(current):
            base = _rand_primitive_unit()
            curA = max(self._area(current), 1e-9)
            tgt_frac = np.random.uniform(*partner_area_frac_range)
            tgtA = curA * tgt_frac
            baseA = max(self._area(base), 1e-9)
            s = np.sqrt(tgtA / baseA)
            partner = scale(base, xfact=s, yfact=s, origin='center')
            partner = self._recenter_and_fit_if_needed(self._clean(partner))

            # place to achieve overlap within range
            for _ in range(12):
                ang = np.random.uniform(0, 2*np.pi)
                rad = np.random.uniform(0.0, 0.6)
                dx, dy = rad*np.cos(ang), rad*np.sin(ang)
                candidate = translate(partner, xoff=dx, yoff=dy)
                candidate = rotate(candidate, angle=np.random.uniform(*self.unit_jitter_rotate_range), origin='center')
                candidate = self._recenter_and_fit_if_needed(self._clean(candidate))
                ov = _overlap_ratio(current, candidate)
                if min_ov <= ov <= max_ov:
                    return candidate
            return self._recenter_and_fit_if_needed(self._clean(partner))

        # seed
        current = self._to_unit_fit(np.random.choice(self.shape_primitives))
        for _ in range(10):
            if self._is_connected_poly(current):
                break
            current = self._to_unit_fit(np.random.choice(self.shape_primitives))
        if not self._is_connected_poly(current):
            current = self._clean(self._to_unit_fit(self.ps.square()).union(self._to_unit_fit(self.ps.circle()))).convex_hull
            current = self._recenter_and_fit_if_needed(current)

        # plan
        steps = int(np.random.randint(max(binary_op_num, min_steps), max_steps + 1))
        plan = ['binary'] * binary_op_num + ['unary'] * (steps - binary_op_num)
        np.random.shuffle(plan)

        executed_binary = 0
        for typ in plan:
            updated = None
            if typ == 'unary':
                for _try in range(8):
                    t = np.random.choice(['scale', 'rotate', 'translate'])
                    if t == 'scale':
                        sx = np.random.uniform(*self.unit_step_scale_range)
                        sy = np.random.uniform(*self.unit_step_scale_range)
                        cand = scale(current, xfact=sx, yfact=sy, origin='center')
                    elif t == 'rotate':
                        ang = np.random.uniform(*self.unit_step_rotate_range)
                        cand = rotate(current, angle=ang, origin='center')
                    else:
                        dx = np.random.uniform(*self.unit_step_translate_range)
                        dy = np.random.uniform(*self.unit_step_translate_range)
                        cand = translate(current, xoff=dx, yoff=dy)

                    cand = self._recenter_and_fit_if_needed(self._clean(cand))
                    if self._is_valid_nonempty(cand) and self._is_connected_poly(cand) and not cand.equals(current):
                        updated = cand
                        break
                if updated is None:
                    updated = self._recenter_and_fit_if_needed(self._clean(current.convex_hull))
            else:
                for _try in range(14):
                    partner = _warp_partner_for_current(current)
                    op_name = np.random.choice(binary_ops_menu)

                    ov = _overlap_ratio(current, partner)
                    if op_name == 'intersect' and ov < min_ov:
                        op_name = 'union'
                    if op_name == 'subtract' and ov < 0.15:
                        op_name = 'xor'

                    candidate = self.apply_binary_op(current, partner, op_name)
                    candidate = _connectedify(candidate, current)
                    candidate = self._recenter_and_fit_if_needed(self._clean(candidate))

                    if not (self._is_valid_nonempty(candidate) and self._is_connected_poly(candidate) and not candidate.equals(current)):
                        continue

                    chg = _symmetric_diff_ratio(current, candidate)
                    if not (min_chg <= chg <= max_chg):
                        continue

                    if _almost_equal(candidate, partner, tol=0.95):
                        continue

                    if op_name in ['subtract', 'xor']:
                        if self._area(candidate) < destructive_min_area_frac * self._area(current):
                            continue

                    updated = candidate
                    executed_binary += 1
                    break

                if updated is None:
                    partner = _warp_partner_for_current(current)
                    candidate = self.apply_binary_op(current, partner, 'union')
                    candidate = _connectedify(candidate, current)
                    candidate = self._recenter_and_fit_if_needed(self._clean(candidate))
                    if self._is_valid_nonempty(candidate) and self._is_connected_poly(candidate) and not candidate.equals(current):
                        updated = candidate
                        executed_binary += 1
                    else:
                        plan.append('binary')
                        continue

            current = updated if updated is not None else current

        while executed_binary < binary_op_num:
            partner = _warp_partner_for_current(current)
            cand = self.apply_binary_op(current, partner, 'union')
            cand = _connectedify(cand, current)
            cand = self._recenter_and_fit_if_needed(self._clean(cand))
            if self._is_valid_nonempty(cand) and self._is_connected_poly(cand) and not cand.equals(current):
                current = cand
                executed_binary += 1

        return self._from_unit(current)

    # ---------- Canvas <-> Unit ----------
    def _from_unit(self, g):
        S = float(self.config['COMPLEX_SHAPE_GEN']['canvas_size'])
        g = scale(g, xfact=S/2.0, yfact=S/2.0, origin=(0.0, 0.0))
        g = translate(g, xoff=S/2.0, yoff=S/2.0)
        return self._clean(g)

    # ---------- Public API ----------
    def create_shape_one_depth(self, depth, gen_num):
        assert depth >= 0
        assert gen_num >= 1

        save_dir = self.config['COMPLEX_SHAPE_GEN'].get('shape_save_dir', 'shape_corpus')
        os.makedirs(save_dir, exist_ok=True)
        save_dir = os.path.join(save_dir, 'depth_{}'.format(depth))
        os.makedirs(save_dir, exist_ok=True)

        shape_dict = dict(depth=depth, shapes=[])
        for num_id in range(gen_num):
            min_step_num = depth
            max_step_num = depth + 4
            binary_op_num = depth
            shape = self.create_shapes(binary_op_num, min_step_num, max_step_num)
            assert (not shape.is_empty) and shape.is_valid and shape.geom_type == 'Polygon', \
                "Generated shape must be a valid non-empty Polygon"

            plot_utils.plot_one_shape(shape, save_path=os.path.join(save_dir, 'gened_shape_{}.svg'.format(num_id)))
            shape_info = self.shape_to_masks_and_pose(shape)
            shape_dict['shapes'].append({'shape': shape, 'shape_info': shape_info})
        
        #save the shape_dict using pickle
        with open(os.path.join(save_dir, 'shape_dict.pkl'), 'wb') as f:
            pickle.dump(shape_dict, f)

        return shape_dict

    # ---------- Pose & raster ----------
    def to_unit_and_pose(self, geom_canvas):
        """
        Decompose a canvas-space geometry g into:
        unit_geom  (fitted to unit disk, NOT clipped),
        pose = {tx, ty, s_recon}, where:
            g ≈ translate( scale(unit_geom, s_recon, s_recon, origin=(0,0)), tx, ty )

        Also returns a pose_matrix and normalized pose_vector for convenience.
        """
        S = float(self.config['COMPLEX_SHAPE_GEN']['canvas_size'])
        fit_margin = 1e-3  # MUST match the margin used in _to_unit_fit

        g = geom_canvas

        # 1) translation = centroid
        cx, cy = list(g.centroid.coords)[0]
        tx, ty = cx, cy

        # 2) center at origin
        g0 = translate(g, xoff=-tx, yoff=-ty)

        # 3) compute the radius bound BEFORE fitting (in canvas units)
        r = self._max_radius_about_origin(g0)
        if r <= 0:
            r = 1.0

        # 4) fit to unit frame (no clipping)
        unit_geom = self._to_unit_fit(g0, margin=fit_margin)

        # 5) pose info
        pose_translate_range = [self.config['COMPLEX_SHAPE_GEN']['pose_gen_translate_range'][0],
                                self.config['COMPLEX_SHAPE_GEN']['pose_gen_translate_range'][1]]
        pose_scale_range = [self.config['COMPLEX_SHAPE_GEN']['pose_scale_range'][0],
                            self.config['COMPLEX_SHAPE_GEN']['pose_scale_range'][1]]
        
        pose_tx = np.random.uniform(pose_translate_range[0], pose_translate_range[1])
        pose_ty = np.random.uniform(pose_translate_range[0], pose_translate_range[1])
        pose_scale = np.random.uniform(pose_scale_range[0], pose_scale_range[1])

        pose_vec = [pose_tx, pose_ty, pose_scale]
        pose_vec_norm = np.array(pose_vec, dtype=np.float32)/np.sqrt(np.sum(np.array(pose_vec, dtype=np.float32)**2))
        pose_vec_norm = pose_vec_norm.tolist()

        pose = {'tx': pose_tx, 'ty': pose_ty, 'scale': pose_scale, 'pose_norm': pose_vec_norm}

        return unit_geom, pose

    def rasterize_unit_cart(self, unit_geom, res=300, ss=1):
        mask = np.zeros((res, res), dtype=np.float32)
        if unit_geom.is_empty:
            return mask.astype(np.uint8)

        gu = prep(unit_geom)
        xs = (np.arange(res) + 0.5) / res * 2.0 - 1.0
        ys = (np.arange(res) + 0.5) / res * 2.0 - 1.0

        if ss <= 1:
            for j, y in enumerate(ys[::-1]):
                row_pts = [Point(x, y) for x in xs]
                mask[res - 1 - j, :] = [gu.contains(p) or gu.touches(p) for p in row_pts]
        else:
            offs = (np.arange(ss) + 0.5) / (ss * res) * 2.0 - 1.0 / res
            for j in range(res):
                y0 = ys[::-1][j]
                vals = []
                for i in range(res):
                    x0 = xs[i]
                    cnt = 0
                    for oy in offs:
                        for ox in offs:
                            p = Point(x0 + ox, y0 + oy)
                            if gu.contains(p) or gu.touches(p):
                                cnt += 1
                    vals.append(cnt / (ss * ss))
                mask[res - 1 - j, :] = vals

        return (mask >= 0.5).astype(np.uint8)

    def rasterize_unit_polar(self, unit_geom, r_bins=300, theta_bins=300, ss=1):
        M = np.zeros((r_bins, theta_bins), dtype=np.float32)
        if unit_geom.is_empty:
            return M.astype(np.uint8)

        gu = prep(unit_geom)
        rs = (np.arange(r_bins) + 0.5) / r_bins
        thetas = (np.arange(theta_bins) + 0.5) / theta_bins * (2.0 * np.pi)

        if ss <= 1:
            for ri, r in enumerate(rs):
                for ti, th in enumerate(thetas):
                    x = r * np.cos(th)
                    y = r * np.sin(th)
                    p = Point(x, y)
                    M[ri, ti] = 1.0 if (gu.contains(p) or gu.touches(p)) else 0.0
        else:
            dr = 1.0 / r_bins
            dth = 2.0 * np.pi / theta_bins
            jr = (np.arange(ss) + 0.5) / (ss) - 0.5
            jth = (np.arange(ss) + 0.5) / (ss) - 0.5
            for ri, r in enumerate(rs):
                for ti, th in enumerate(thetas):
                    cnt = 0
                    for a in jr:
                        for b in jth:
                            rj = max(0.0, min(1.0, r + a * dr))
                            thj = th + b * dth
                            x = rj * np.cos(thj)
                            y = rj * np.sin(thj)
                            if gu.contains(Point(x, y)) or gu.touches(Point(x, y)):
                                cnt += 1
                    M[ri, ti] = cnt / (ss * ss)

        return (M >= 0.5).astype(np.uint8)

    def shape_to_masks_and_pose(self, geom_canvas, cart_res=300, r_bins=300, theta_bins=300, ss=1):
        unit_geom, pose = self.to_unit_and_pose(geom_canvas)
        cart_mask = self.rasterize_unit_cart(unit_geom, res=cart_res, ss=ss)
        polar_mask = self.rasterize_unit_polar(unit_geom, r_bins=r_bins, theta_bins=theta_bins, ss=ss)

        return {
            "unit_geom": unit_geom,
            "cart_mask": cart_mask,
            "polar_mask": polar_mask,
            "pose": pose,
        }