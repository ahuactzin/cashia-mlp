MLP_RESOURCE_KEYS = {
    "inflation_table": "mlp/inputs/tablaDeInflacion.csv"
}

def get_mlp_resource_key(name: str) -> str:
    try:
        return MLP_RESOURCE_KEYS[name]
    except KeyError as e:
        available = ", ".join(sorted(MLP_RESOURCE_KEYS.keys()))
        raise KeyError(
            f"Unknown MLP resource: '{name}'. Available: {available}"
        ) from e