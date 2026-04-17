
from shapely.affinity import scale, rotate, translate
from shapely.ops import unary_union
    
class UnaryShapeOp:
    def __init__(self):
        self.op_name = 'UnaryShapeOp'

    def run_op(self, shape, op_name, **kwargs):
        assert op_name in ['buffer', 'simplify', 'convex_hull', 'boundary', 'centroid'], f"Unknown operation: {op_name}"
        if op_name == 'buffer':
            return shape.buffer(kwargs.get('distance', 5))
        elif op_name == 'simplify':
            return shape.simplify(kwargs.get('tolerance', 2))
        elif op_name == 'convex_hull':
            return shape.convex_hull
        elif op_name == 'boundary':
            return shape.boundary
        elif op_name == 'centroid':
            return shape.centroid
        
        return shape
    
class UnaryShapeTransform:
    def __init__(self):
        self.name = 'UnaryShapeTransform'
    
    def run_transform(self, shape, op_name, **kwargs):
        assert op_name in ['scale', 'rotate', 'translate'], f"Unknown operation: {op_name}"
        if op_name == 'scale':
            return scale(shape, kwargs.get('scale_x', 1), kwargs.get('scale_y', 1), origin='center')
        elif op_name == 'rotate':
            return rotate(shape, kwargs.get('angle', 0), origin='center')
        elif op_name == 'translate':
            return translate(shape, kwargs.get('x', 0), kwargs.get('y', 0))
        
        return shape


class BinaryShapeOp:
    def __init__(self):
        self.name = 'BinaryShapeOp'

    def run_op(self, shape1, shape2, op_name):
        assert op_name in ['union', 'intersect', 'subtract', 'xor', 'convex_hull'], f"Unknown operation: {op_name}"
        if op_name == 'union':
            return unary_union([shape1, shape2])
        elif op_name == 'intersect':
            return shape1.intersection(shape2)
        elif op_name == 'subtract':
            return shape1.difference(shape2)
        elif op_name == 'xor':
            return shape1.symmetric_difference(shape2)
        elif op_name == 'convex_hull':
            return unary_union([shape1, shape2]).convex_hull
        else:
            raise ValueError(f"Unknown operation: {op_name}")