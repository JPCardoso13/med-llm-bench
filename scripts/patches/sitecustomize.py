# Patches prometheus_fastapi_instrumentator 8.x to handle FastAPI's _IncludedRouter
# objects that lack a 'path' attribute, causing an AttributeError that makes every
# HTTP request to vLLM return 500.  Injected via PYTHONPATH by vllm_launcher.py.
try:
    from starlette.routing import Match, Mount
    import prometheus_fastapi_instrumentator.routing as _pfr

    def _fixed_get_route_name(scope, routes, route_name=None):
        for route in routes:
            if not hasattr(route, "matches"):
                continue
            match, child_scope = route.matches(scope)
            if match == Match.FULL:
                if not hasattr(route, "path"):
                    continue
                route_name = route.path
                child_scope = {**scope, **child_scope}
                if isinstance(route, Mount) and route.routes:
                    child_route_name = _fixed_get_route_name(
                        child_scope, route.routes, route_name
                    )
                    route_name = (
                        None if child_route_name is None else route_name + child_route_name
                    )
                return route_name
            elif match == Match.PARTIAL and route_name is None:
                if hasattr(route, "path"):
                    route_name = route.path
        return None

    _pfr._get_route_name = _fixed_get_route_name
except Exception:
    pass
