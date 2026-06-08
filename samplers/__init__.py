
VERSION = "v1.0.0"  # UPS v1.5.8


# ==============================================================================
# == Unit Tests (smoke — runs without ComfyUI)
# ==============================================================================

if __name__ == "__main__":
    print("\n🧪 Smoke tests: __init__")
    print("   VERSION :", VERSION)
    _pass = _fail = 0

    def _check(label, expr):
        global _pass, _fail
        if expr:
            print(f"  ✅  {label}")
            _pass += 1
        else:
            print(f"  ❌  {label}")
            _fail += 1

    _check("VERSION defined",    VERSION == "v1.0.0")
    _check("NODE_CLASS_MAPPINGS is dict",
           isinstance(NODE_CLASS_MAPPINGS, dict))
    _check("NODE_CLASS_MAPPINGS not empty",
           len(NODE_CLASS_MAPPINGS) > 0)

    print(f"\n  {_pass} passed, {_fail} failed")
    if _fail == 0:
        print("  🎉 All good.")
